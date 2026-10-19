"""The push-to-talk loop: hotkey -> record -> transcribe -> stream LLM -> speak sentence by sentence."""
import contextlib
import logging
import queue
import threading
import time
from pathlib import Path

import numpy as np

from . import cuda_pack, gpu, settings_schema
from .audio import MIC_RATE, Player, Recorder, start_beep, stop_beep
from .config import Config, app_dir
from .llm import CLOUD_BACKENDS, create as create_chat
from .stt import Transcriber
from .tts import SentenceSplitter, Speaker

log = logging.getLogger(__name__)

MIN_SPEECH_S = 0.3
SILENT_LEVEL = 0.005  # peak below this means the mic isn't picking anything up


def activity_of(recording: bool, jobs_busy: bool, sentences_busy: bool, playing: bool) -> str:
    """Pure: what Parley is doing, for the tray icon. listening > speaking > thinking > idle."""
    if recording:
        return "listening"
    if playing:
        return "speaking"
    if jobs_busy or sentences_busy:
        return "thinking"
    return "idle"


class Parley:
    def __init__(self, cfg: Config, use_mic: bool = True, on_pack_ready=None):
        self.cfg = cfg
        self.turn = 0  # bumped whenever a new question starts; stale work checks it and stops
        self._jobs: queue.Queue[tuple[int, np.ndarray | str, float]] = queue.Queue()
        self._sentences: queue.Queue[tuple[int, str | None, float]] = queue.Queue()
        self.finished = threading.Event()  # set once a turn's answer has been fully played
        self._init_apply_state()

        home = app_dir()
        log.info("Hardware: %s", gpu.describe())
        if use_mic:  # a typed --ask smoke run never starts a 1.6 GB download
            cuda_pack.start_background(cfg, gpu.current_plan(), home, self.say, on_pack_ready)  # packaged NVIDIA copies only
        # The LLM loads into VRAM in the background while speech models load
        llm_ready = threading.Thread(target=self._start_llm, args=(home,), daemon=True)
        llm_ready.start()
        self.stt = Transcriber(cfg.stt, home / "models" / "whisper") if use_mic else None
        self.speaker = Speaker(cfg.tts, home / "models" / "kokoro")
        self.player = Player(cfg.audio.output_device, is_current=lambda t: t == self.turn)
        self.recorder = Recorder(cfg.audio.input_device, cfg.audio.preroll_ms) if use_mic else None
        llm_ready.join()
        if self._llm_error:
            raise self._llm_error

        threading.Thread(target=self._think_loop, daemon=True, name="think").start()
        threading.Thread(target=self._speak_loop, daemon=True, name="speak").start()

    def _start_llm(self, home: Path) -> None:
        self._llm_error = None
        try:
            llm = self.cfg.llm
            base_url, local_model = None, "local"
            if llm.backend == "builtin":
                from . import llama_server
                base_url, local_model = llama_server.start(llm, home)
            elif llm.backend in CLOUD_BACKENDS:
                log.info("Using %s: what you say is transcribed locally, then the text is sent to %s",
                         CLOUD_BACKENDS[llm.backend], CLOUD_BACKENDS[llm.backend])
            self.chat = create_chat(llm, self.cfg.tools, base_url, local_model)
            self.chat.warm_up()
        except BaseException as e:  # incl. SystemExit for config errors, re-raised on the main thread
            self._llm_error = e

    def _init_apply_state(self) -> None:
        """State for the settings window (also used directly by tests that build a Parley without loading models)."""
        self.hotkey = None  # the HotkeyListener, set by whoever runs it (tray mode, console loop); Settings re-points it
        self._previewing = False  # a voice sample is being synthesized (counts as busy)
        self._apply_lock = threading.Lock()
        self._restart_paths: set[str] = set()  # saved settings that only a restart applies
        self.deferred: list[str] = []  # settings waiting for the current answer to finish
        self._pending: dict = {"hotkey": None, "audio": {}}  # wanted hotkey / audio-device changes not applied yet
        self.on_applied = None  # called with the config after a deferred change was applied (the tray updates its key name)
        self._watching = False  # the thread that applies them after the answer is running

    def models_in_use(self) -> set[str]:
        """Catalog ids of the models this app is running (or, before they are loaded, would run): the Models page uses it
        to refuse deleting them. The Whisper name comes from the loaded Transcriber, else from the config."""
        from . import models, stt as stt_mod
        name = getattr(getattr(self, "stt", None), "name", None)
        if name is None:
            name = stt_mod.resolve_name(self.cfg.stt, gpu.current_plan().stt == "cuda")
        return models.ids_in_use(self.cfg, app_dir(), name)

    def is_current(self, turn: int) -> bool:
        return turn == self.turn

    def is_busy(self) -> bool:
        """True while a question is being recorded, transcribed, answered or spoken (used to hold back updates).
        The queues' unfinished counts also cover an item a worker has taken but not finished."""
        recording = self.recorder is not None and self.recorder.recording
        return (recording or getattr(self, "_previewing", False)
                or any(q.unfinished_tasks for q in (self._jobs, self._sentences, self.player.queue)))

    def activity(self) -> str:
        """idle | listening | thinking | speaking (cheap, safe to poll from any thread). The tray debounces it, since
        the 60 ms beeps make the player queue busy for an instant."""
        return activity_of(self.recorder is not None and self.recorder.recording,
                           bool(self._jobs.unfinished_tasks), bool(self._sentences.unfinished_tasks),
                           bool(self.player.queue.unfinished_tasks))

    def say(self, text: str) -> None:
        """Speak a short notice of our own (e.g. about updates). Dropped if the user starts a question meanwhile."""
        self._sentences.put((self.turn, text, time.perf_counter()))

    def speak_notice(self, sentences: list[str]) -> int:
        """Speak a few sentences of our own (the first-run welcome) as one turn and return its id. `finished` is set
        when the last one has really been played; a key press bumps `turn`, drops the rest and leaves it unset."""
        turn, t0 = self.turn, time.perf_counter()
        self.finished.clear()
        for sentence in sentences:
            self._sentences.put((turn, sentence, t0))
        self._sentences.put((turn, None, t0))
        return turn

    # --- settings window (any thread) ---

    def voices(self) -> list[str]:
        """Voice names the installed Kokoro model offers."""
        return sorted(self.speaker.kokoro.get_voices())

    def preview_voice(self, voice: str, speed: float, text: str) -> bool:
        """Say `text` in `voice` at `speed` without saving anything. False if Parley is busy; raises ValueError for a
        voice or speed that isn't acceptable. Runs on a worker thread; a hotkey press cuts it off (turn counter)."""
        speed = settings_schema.validate("tts.speed", speed)
        if voice not in self.voices():
            raise ValueError("That voice isn't installed.")
        text = text.strip()[:300]
        if not text:
            return False
        with self._apply_lock:
            if self.is_busy():
                return False
            self._previewing = True
        turn = self.turn  # a key press bumps it, which drops the sample at once

        def work() -> None:
            try:
                audio = self.speaker.synth(text, voice=voice, speed=speed,
                                           lang=settings_schema.voice_lang(voice) or self.cfg.tts.lang)
                if self.is_current(turn):
                    self.player.play(turn, audio)
            except Exception:
                log.exception("Voice sample failed")
            finally:
                self._previewing = False  # the clip is queued by now: the player's queue keeps is_busy() true

        threading.Thread(target=work, daemon=True, name="voice-sample").start()
        return True

    def restart_pending(self) -> list[str]:
        """Setting paths whose saved value only a restart will apply."""
        return sorted(self._restart_paths)

    def apply_config(self, new: Config) -> tuple[list[str], list[str]]:
        """Make a saved config take effect. Returns (applied, needs_restart) as setting paths.

        The shared sub-dataclasses are changed IN PLACE (Speaker, Chat and the providers hold references to them), so
        `live` settings are seen at their next use. `hotkey` and `swap_audio` settings are applied while Parley is idle;
        during an answer they wait (`self.deferred`) and apply right after it. Restart-only settings (and the ones
        whose machinery lands in a later change: the language-model backend and tools) are NOT written into the running
        config, so the difference stays visible and Settings keeps offering "Restart now". `llm.api_key` is never handled
        here: keys live in the credential store only.
        """
        S = settings_schema
        applied, restart = [], []
        hotkey_new, audio_new = None, {}
        with self._apply_lock:
            for s in S.SETTINGS:
                if s.path == "llm.api_key":
                    continue
                table, key = s.path.split(".")
                old_v, new_v = getattr(getattr(self.cfg, table), key), getattr(getattr(new, table), key)
                if old_v == new_v:
                    self._restart_paths.discard(s.path)  # edited back to what is running
                    continue
                if s.apply in (S.LIVE, S.NEXT_LAUNCH):
                    setattr(getattr(self.cfg, table), key, new_v)
                    applied.append(s.path)
                elif s.apply == S.HOTKEY:
                    hotkey_new = new_v
                elif s.apply == S.SWAP_AUDIO:
                    audio_new[s.path] = new_v
                else:  # RESTART, and REBUILD_CHAT / RELOAD_LLM until their machinery exists
                    self._restart_paths.add(s.path)
                    restart.append(s.path)
            self._pending = {"hotkey": hotkey_new, "audio": audio_new}  # replaces an older wait: this is what's wanted now
            a, r = self._apply_idle_changes()
        return applied + a, restart + r

    def _apply_idle_changes(self) -> tuple[list[str], list[str]]:
        """Apply the wanted hotkey / audio-device changes if Parley is idle, else wait for the end of the answer
        (`self.deferred` names them). Call with `_apply_lock` held."""
        hotkey_new, audio_new = self._pending["hotkey"], self._pending["audio"]
        if not hotkey_new and not audio_new:
            self.deferred = []
            return [], []
        if self.is_busy():
            self.deferred = (["hotkey.key"] if hotkey_new else []) + sorted(audio_new)
            if not self._watching:
                self._watching = True
                threading.Thread(target=self._apply_when_idle, daemon=True, name="apply-config").start()
            return [], []
        applied, restart = [], []
        if hotkey_new:
            try:
                if self.hotkey is None:
                    raise SystemExit("no hotkey listener")
                self.hotkey.set_key(hotkey_new)
                self.cfg.hotkey.key = hotkey_new
                applied.append("hotkey.key")
            except SystemExit as e:
                log.warning("Hotkey not changed (%s); restart Parley to use it", e)
                restart.append("hotkey.key")
        if audio_new:
            if self._swap_audio(audio_new):
                applied += sorted(audio_new)
            else:
                restart += sorted(audio_new)
        self._restart_paths.update(restart)
        self._pending, self.deferred = {"hotkey": None, "audio": {}}, []
        return applied, restart

    def _apply_when_idle(self) -> None:
        while True:
            time.sleep(0.25)
            with self._apply_lock:
                if not self.deferred:  # a later save replaced or cleared the wait
                    self._watching = False
                    return
                if self.is_busy():
                    continue
                applied, restart = self._apply_idle_changes()
                if self.deferred:  # went busy again between the check and the apply: keep watching
                    continue
                self._watching = False
            log.info("Applied after the answer: %s%s", applied, f"; restart needed for {restart}" if restart else "")
            if self.on_applied is not None:
                try:
                    self.on_applied(self.cfg)
                except Exception:
                    log.debug("on_applied failed", exc_info=True)
            return

    def _make_recorder(self, device, preroll_ms: int) -> Recorder:
        return Recorder(device, preroll_ms)

    def _make_player(self, device) -> Player:
        return Player(device, is_current=lambda t: t == self.turn)

    def _swap_audio(self, changes: dict) -> bool:
        """Build the new microphone / speaker first; only when that worked replace the old ones and close them.
        On any failure the old devices keep running and False is returned (the caller reports a restart)."""
        audio = self.cfg.audio
        rec = player = None
        try:
            if self.recorder is not None and ({"audio.input_device", "audio.preroll_ms"} & changes.keys()):
                rec = self._make_recorder(changes.get("audio.input_device", audio.input_device),
                                          changes.get("audio.preroll_ms", audio.preroll_ms))
            if "audio.output_device" in changes:
                player = self._make_player(changes["audio.output_device"])
        except Exception as e:
            log.warning("Could not switch the audio device (%s); restart Parley to use it", e)
            for made in (rec, player):
                if made is not None:
                    try:
                        made.close()
                    except Exception:
                        pass
            return False
        old = []
        if rec is not None:
            old.append(self.recorder)
            self.recorder = rec
        if player is not None:
            old.append(self.player)
            self.player = player
        for o in old:
            try:
                o.close()
            except Exception:
                log.debug("Closing the old audio device failed", exc_info=True)
        for path, value in changes.items():
            setattr(audio, path.split(".")[1], value)
        return True

    # --- hotkey events (called from the keyboard hook thread; keep them quick) ---

    def begin_listening(self) -> None:
        if self.recorder.recording:
            return  # key auto-repeat
        self.turn += 1  # cuts off anything still being said or generated
        self.recorder.start()
        if self.cfg.audio.sounds:
            self.player.play(self.turn, start_beep())

    def end_listening(self) -> None:
        if not self.recorder.recording:
            return
        audio = self.recorder.stop()
        if self.cfg.audio.sounds:
            self.player.play(self.turn, stop_beep())
        self._jobs.put((self.turn, audio, time.perf_counter()))

    def ask(self, text: str) -> int:
        """Submit a typed question (no mic). Returns its turn id."""
        self.turn += 1
        self.finished.clear()
        self._jobs.put((self.turn, text, time.perf_counter()))
        return self.turn

    # --- workers ---

    def _think_loop(self) -> None:
        while True:
            turn, item, t0 = self._jobs.get()
            try:
                if not self.is_current(turn):
                    continue
                try:
                    self._think(turn, item, t0)
                except Exception:
                    log.exception("Failed to answer")
                    self._sentences.put((turn, "Sorry, something went wrong.", t0))
                self._sentences.put((turn, None, t0))  # end of turn
            finally:
                self._jobs.task_done()  # after the follow-up puts, so is_busy() never sees a gap

    def _think(self, turn: int, item: np.ndarray | str, t0: float) -> None:
        if isinstance(item, str):
            text = item
        else:
            level = float(np.abs(item).max()) if len(item) else 0.0
            log.info("Recorded %.1fs, peak level %.0f%%", len(item) / MIC_RATE, level * 100)
            if len(item) < MIN_SPEECH_S * MIC_RATE:
                return  # a tap of the key, not a question
            if level < SILENT_LEVEL:
                log.warning("The microphone is silent. Run `parley --test-mic` to check which one is being used.")
                self._sentences.put((turn, "I couldn't hear anything. Check which microphone I'm using.", t0))
                return
            text = self.stt.transcribe(item)
            log.info("You: %s  [stt %.0f ms]", text or "(nothing recognised)", (time.perf_counter() - t0) * 1000)
            if not text:
                self._sentences.put((turn, "Sorry, I didn't catch that.", t0))
                return
        splitter = SentenceSplitter()
        first_token = None
        answer = []
        with contextlib.closing(self.chat.reply(text, cancelled=lambda: not self.is_current(turn))) as pieces:
            for piece in pieces:
                if first_token is None:
                    first_token = time.perf_counter()
                    log.debug("first token %.0f ms", (first_token - t0) * 1000)
                answer.append(piece)
                for sentence in splitter.feed(piece):
                    self._sentences.put((turn, sentence, t0))
        if self.is_current(turn):
            for sentence in splitter.flush():
                self._sentences.put((turn, sentence, t0))
            log.info("Parley: %s", " ".join("".join(answer).split()))

    def _speak_loop(self) -> None:
        first = True
        while True:
            turn, sentence, t0 = self._sentences.get()
            try:
                if sentence is None:
                    first = True
                    self.player.play(turn, np.zeros(0, dtype=np.float32), self.finished.set)
                    continue
                if not self.is_current(turn):
                    continue
                t = time.perf_counter()
                audio = self.speaker.synth(sentence)
                log.debug("tts %.0f ms (+%.0f ms since release): %s", (time.perf_counter() - t) * 1000, (time.perf_counter() - t0) * 1000, sentence)
                on_start = None
                if first:
                    first = False
                    on_start = lambda t0=t0: log.info("[first audio %.0f ms after release]", (time.perf_counter() - t0) * 1000)
                self.player.play(turn, audio, on_start)
            finally:
                self._sentences.task_done()
