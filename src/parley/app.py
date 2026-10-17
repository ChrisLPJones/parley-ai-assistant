"""The push-to-talk loop: hotkey -> record -> transcribe -> stream LLM -> speak sentence by sentence."""
import contextlib
import logging
import queue
import threading
import time
from pathlib import Path

import numpy as np

from . import cuda_pack, gpu
from .audio import MIC_RATE, Player, Recorder, tone
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
    def __init__(self, cfg: Config, use_mic: bool = True):
        self.cfg = cfg
        self.turn = 0  # bumped whenever a new question starts; stale work checks it and stops
        self._jobs: queue.Queue[tuple[int, np.ndarray | str, float]] = queue.Queue()
        self._sentences: queue.Queue[tuple[int, str | None, float]] = queue.Queue()
        self.finished = threading.Event()  # set once a turn's answer has been fully played

        home = app_dir()
        log.info("Hardware: %s", gpu.describe())
        if use_mic:  # a typed --ask smoke run never starts a 1.6 GB download
            cuda_pack.start_background(cfg, gpu.current_plan(), home, self.say)  # packaged NVIDIA copies only
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

    def is_current(self, turn: int) -> bool:
        return turn == self.turn

    def is_busy(self) -> bool:
        """True while a question is being recorded, transcribed, answered or spoken (used to hold back updates).
        The queues' unfinished counts also cover an item a worker has taken but not finished."""
        recording = self.recorder is not None and self.recorder.recording
        return recording or any(q.unfinished_tasks for q in (self._jobs, self._sentences, self.player.queue))

    def activity(self) -> str:
        """idle | listening | thinking | speaking (cheap, safe to poll from any thread). The tray debounces it, since
        the 60 ms beeps make the player queue busy for an instant."""
        return activity_of(self.recorder is not None and self.recorder.recording,
                           bool(self._jobs.unfinished_tasks), bool(self._sentences.unfinished_tasks),
                           bool(self.player.queue.unfinished_tasks))

    def say(self, text: str) -> None:
        """Speak a short notice of our own (e.g. about updates). Dropped if the user starts a question meanwhile."""
        self._sentences.put((self.turn, text, time.perf_counter()))

    # --- hotkey events (called from the keyboard hook thread; keep them quick) ---

    def begin_listening(self) -> None:
        if self.recorder.recording:
            return  # key auto-repeat
        self.turn += 1  # cuts off anything still being said or generated
        self.recorder.start()
        if self.cfg.audio.sounds:
            self.player.play(self.turn, tone(880))

    def end_listening(self) -> None:
        if not self.recorder.recording:
            return
        audio = self.recorder.stop()
        if self.cfg.audio.sounds:
            self.player.play(self.turn, tone(660))
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
