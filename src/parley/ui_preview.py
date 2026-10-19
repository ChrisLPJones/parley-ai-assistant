"""Open the REAL settings window against a stub app and a throwaway home: no models, no microphone, no audio.

    python -m parley.ui_preview [--theme light|dark] [--scale 100|125|150|200] [--page general|voice|models]
                                [--state restart-needed|error|downloading|download-error] [--auto-close SECONDS] [--smoke]

For looking at the window (and for scripts/shot_window.py). `PARLEY_HOME` is forced to a temp folder named
`parley_test_ui_preview_*`, so Save writes a real config.toml there and nothing else is touched; the autostart control
uses an in-memory fake. The Models page runs against `StubModels` (fake files, a simulated download, no network).
`--smoke` opens the window, checks it exists and exits 0.
"""
import argparse
import logging
import os
import shutil
import sys
import tempfile
import threading
import time
from pathlib import Path

HOME_PREFIX = "parley_test_"
KOKORO_VOICES = (
    "af_alloy af_aoede af_bella af_heart af_jessica af_kore af_nicole af_nova af_river af_sarah af_sky "
    "am_adam am_echo am_eric am_fenrir am_liam am_michael am_onyx am_puck am_santa "
    "bf_alice bf_emma bf_isabella bf_lily bm_daniel bm_fable bm_george bm_lewis ef_dora em_alex em_santa ff_siwis "
    "hf_alpha hf_beta hm_omega hm_psi if_sara im_nicola jf_alpha jf_gongitsune jf_nezumi jf_tebukuro jm_kumo "
    "pf_dora pm_alex pm_santa zf_xiaobei zf_xiaoni zf_xiaoxiao zf_xiaoyi zm_yunjian zm_yunxi zm_yunxia zm_yunyang").split()
CONFIG_TEXT = "# Parley settings (preview)\n[tts]\nvoice = \"af_heart\"  # my favourite\nspeed = 1.1\n"


def throwaway_home() -> tuple[Path, bool]:
    """(home, created_by_us). Reuses PARLEY_HOME only if it is already a `parley_test_*` folder; never the real home."""
    env = os.environ.get("PARLEY_HOME")
    if env and Path(env).name.startswith(HOME_PREFIX):
        Path(env).mkdir(parents=True, exist_ok=True)
        return Path(env), False
    return Path(tempfile.mkdtemp(prefix=HOME_PREFIX + "ui_preview_")), True


class FakeAutostart:
    def __init__(self, enabled: bool = False, available: bool = True) -> None:
        from . import autostart

        self._state = autostart.State(available, enabled, "" if available else "Only available in the installed copy of Parley.")

    def status(self):
        return self._state

    def set_enabled(self, on: bool) -> None:
        from . import autostart

        self._state = autostart.State(self._state.available, on)


class FakeHotkey:
    """Stands in for the key listener: 'pressing a key' (F13) one second after capture starts."""

    def __init__(self) -> None:
        self._capture = None

    @property
    def capture(self):
        return self._capture

    @capture.setter
    def capture(self, cb) -> None:
        self._capture = cb
        if cb is not None:
            def press() -> None:
                if self._capture is cb:
                    cb(0x7C)
            threading.Timer(1.0, press).start()


class StubModels:
    """Stands in for `models.ModelManager`: the real catalog, fake files on a fake computer, and a download that is only a
    clock (about 40 MB/s). Nothing is read from or written to disk and nothing is fetched."""

    RATE = 40 * 1024**2

    def __init__(self, mode: str | None = None, home: Path | None = None) -> None:
        from . import models

        self.home = home or Path(tempfile.gettempdir())
        self.catalog = models.CATALOG
        self._m = models
        self._lock = threading.Lock()
        self._installed = {"llm-tiny", "llm-small", "stt-small.en", "stt-base.en", "tts-kokoro"}
        self._partial = {"stt-medium": 400 * 1024**2}
        self._in_use = {"llm-tiny", "stt-small.en", "tts-kokoro"}
        self._dl = models.DownloadState()
        self._t0 = 0.0
        if mode == "downloading":  # 43 % of the 9B model
            total = self.catalog["llm-medium"].total_bytes
            self._begin("llm-medium", time.monotonic() - 0.43 * total / self.RATE)
        elif mode == "download-error":
            self._dl = models.DownloadState("llm-medium", 0, 1, 0, 0, "error",
                                            "The Qwen3.5 9B (medium) download failed its safety check, so Parley discarded it.")

    def _begin(self, mid: str, t0: float) -> None:
        m = self.catalog[mid]
        self._t0 = t0
        self._dl = self._m.DownloadState(mid, 0, len(m.files) or 1, 0, m.total_bytes, "downloading")

    @property
    def state(self):
        with self._lock:
            d = self._dl
            if d.phase == "downloading":
                done = min(d.total, int((time.monotonic() - self._t0) * self.RATE))
                if done >= d.total:
                    self._installed.add(d.model_id)
                    self._partial.pop(d.model_id, None)
                    self._dl = self._m.DownloadState(d.model_id, 0, d.file_count, d.total, d.total, "done",
                                                     f"{self.catalog[d.model_id].label} is ready.")
                else:
                    self._dl = self._m.DownloadState(d.model_id, 0, d.file_count, done, d.total, "downloading")
            return self._dl

    def status(self, m):
        st = self.state
        if st.phase == "downloading" and st.model_id == m.id:
            return self._m.Status("downloading", st.done)
        if m.id in self._installed:
            return self._m.Status("installed", m.total_bytes)
        if m.id in self._partial:
            return self._m.Status("partial", self._partial[m.id])
        return self._m.Status("not_installed", 0)

    def scan(self):
        return {mid: self.status(m) for mid, m in self.catalog.items()}

    def in_use_ids(self):
        return set(self._in_use)

    def download(self, model_id: str) -> None:
        m = self.catalog.get(model_id)
        if m is None or not m.pinned:
            raise self._m.ModelError("Parley does not know that model.")
        if self.state.phase == "downloading":
            raise self._m.ModelError("Another model is already downloading. Wait for it or cancel it first.")
        with self._lock:
            self._begin(model_id, time.monotonic() - self._partial.get(model_id, 0) / self.RATE)

    def cancel(self) -> None:
        st = self.state
        if st.phase == "downloading":
            with self._lock:
                self._partial[st.model_id] = st.done
                self._dl = self._m.DownloadState(st.model_id, 0, st.file_count, st.done, st.total, "cancelled",
                                                 "Download cancelled. What was downloaded is kept so you can continue.")

    def delete(self, model_id: str) -> int:
        m = self.catalog.get(model_id)
        if m is None or model_id in self._in_use:
            raise self._m.ModelError("That model is in use. Switch to another model first, then delete it.")
        freed = m.total_bytes if model_id in self._installed else self._partial.get(model_id, 0)
        self._installed.discard(model_id)
        self._partial.pop(model_id, None)
        return freed


def stub_hardware(language: str = "en"):
    """A mid-range laptop: 6 GB graphics card, 12 GB memory, so the 9B model shows as 'too big'."""
    from . import models

    return models.Hardware(vram_mb=6144, gpu_name="NVIDIA GeForce RTX 3060 Laptop GPU", ram_mb=12288, free_disk_mb=120 * 1024,
                           llm_runtime="cuda", speech_runtime="cuda", preset="small", language=language)


class StubApp:
    """The parts of `Parley` the window uses. `apply_config` classifies like the real one (by the schema), nothing runs."""

    def __init__(self, cfg, restart_pending=()) -> None:
        self.cfg = cfg
        self.deferred: list[str] = []
        self._pending = list(restart_pending)
        self._busy_until = 0.0
        self.played: list[tuple[str, float]] = []

    def voices(self) -> list[str]:
        return list(KOKORO_VOICES)

    def is_busy(self) -> bool:
        return time.monotonic() < self._busy_until

    def preview_voice(self, voice: str, speed: float, text: str) -> bool:
        if self.is_busy():
            return False
        self.played.append((voice, speed))
        self._busy_until = time.monotonic() + 2.0  # pretend the clip lasts two seconds; no audio is made
        return True

    def restart_pending(self) -> list[str]:
        return list(self._pending)

    def apply_config(self, new):
        from . import settings_schema as S

        applied, restart = [], []
        for s in S.SETTINGS:
            table, key = s.path.split(".")
            if getattr(getattr(self.cfg, table), key) == getattr(getattr(new, table), key):
                continue
            (restart if s.apply in (S.RESTART, S.REBUILD_CHAT, S.RELOAD_LLM) else applied).append(s.path)
        for p in restart:
            if p not in self._pending:
                self._pending.append(p)
        return applied, restart


def build_host(state: str | None = None, *, home: Path | None = None, models_state: str | None = None):
    """(host, home, created). The stub app is `host.app`."""
    from . import config, settings_window

    created = False
    if home is None:
        home, created = throwaway_home()
    os.environ["PARLEY_HOME"] = str(home)
    cfg_path = home / "config.toml"
    if not cfg_path.exists():
        cfg_path.write_text(CONFIG_TEXT, encoding="utf-8")
    app = StubApp(config.load(cfg_path), ["stt.model", "tts.device"] if state == "restart-needed" else ())
    error = "The speech model could not be loaded. Check stt.model in config.toml." if state == "error" else None
    names = {"input": ["Microphone (USB Audio Device)", "Headset Microphone (Realtek Audio)", "Line In (Realtek Audio)"],
             "output": ["Speakers (Realtek Audio)", "Headphones (USB Audio Device)", "Monitor (HDMI Audio)"]}
    hotkey = FakeHotkey()
    host = settings_window.Host(cfg_path, get_app=lambda: app, get_error=lambda: error,
                                restart=lambda: print("Restart now clicked (preview: nothing restarts)"),
                                get_hotkey=lambda: hotkey, list_devices=lambda kind: names[kind],
                                autostart_api=FakeAutostart(), models_api=StubModels(models_state, home),
                                detect_hardware=stub_hardware)
    return host, home, created


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m parley.ui_preview", description=__doc__.split("\n")[0])
    p.add_argument("--theme", choices=("light", "dark"), help="default: follow Windows")
    p.add_argument("--scale", type=int, choices=(100, 125, 150, 175, 200), help="default: the system's DPI scaling")
    p.add_argument("--page", choices=("general", "voice", "models"), default="general")
    p.add_argument("--state", choices=("restart-needed", "error", "downloading", "download-error"),
                   help="open with that banner showing (models page: a download at 43 %% or a failed one; default downloading)")
    p.add_argument("--auto-close", type=float, metavar="SECONDS", help="close the window after this long")
    p.add_argument("--smoke", action="store_true", help="open the window, check it exists, exit 0")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")

    from . import settings_window, ui

    banner_state = args.state if args.state in ("restart-needed", "error") else None
    models_state = args.state if args.state in ("downloading", "download-error") else ("downloading" if args.page == "models" else None)
    host, home, created = build_host(banner_state, models_state=models_state)
    print(f"ui_preview: throwaway home {home}")
    ui.theme_override, ui.scale_override = args.theme, args.scale
    t0 = time.monotonic()
    try:
        if not settings_window.open_settings(host, args.page):
            print(f"ui_preview: the UI could not start ({ui.last_error()})", file=sys.stderr)
            return 1
        opened = _wait(lambda: ui.call(lambda: settings_window._current is not None and settings_window._current.alive()), 15)
        if not opened:
            print("ui_preview: the window did not open", file=sys.stderr)
            return 1
        print(f"ui_preview: window open after {time.monotonic() - t0:.2f} s")
        if args.smoke:
            return 0
        deadline = time.monotonic() + args.auto_close if args.auto_close else None
        while ui.call(lambda: settings_window._current is not None and settings_window._current.alive()):
            if deadline and time.monotonic() >= deadline:
                break
            time.sleep(0.2)
        return 0
    except KeyboardInterrupt:
        return 0
    finally:
        settings_window.close_settings()
        time.sleep(0.2)
        ui.stop()
        if created:
            shutil.rmtree(home, ignore_errors=True)


def _wait(fn, timeout: float) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if fn():
            return True
        time.sleep(0.05)
    return False


if __name__ == "__main__":
    sys.exit(main())
