"""Open the REAL settings window against a stub app and a throwaway home: no models, no microphone, no audio.

    python -m parley.ui_preview [--theme light|dark] [--scale 100|125|150|200] [--page general|voice]
                                [--state restart-needed|error] [--auto-close SECONDS] [--smoke]

For looking at the window (and for scripts/shot_window.py). `PARLEY_HOME` is forced to a temp folder named
`parley_test_ui_preview_*`, so Save writes a real config.toml there and nothing else is touched; the autostart control
uses an in-memory fake. `--smoke` opens the window, checks it exists and exits 0.
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


def build_host(state: str | None = None, *, home: Path | None = None):
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
                                autostart_api=FakeAutostart())
    return host, home, created


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m parley.ui_preview", description=__doc__.split("\n")[0])
    p.add_argument("--theme", choices=("light", "dark"), help="default: follow Windows")
    p.add_argument("--scale", type=int, choices=(100, 125, 150, 175, 200), help="default: the system's DPI scaling")
    p.add_argument("--page", choices=("general", "voice"), default="general")
    p.add_argument("--state", choices=("restart-needed", "error"), help="open with that banner showing")
    p.add_argument("--auto-close", type=float, metavar="SECONDS", help="close the window after this long")
    p.add_argument("--smoke", action="store_true", help="open the window, check it exists, exit 0")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")

    from . import settings_window, ui

    host, home, created = build_host(args.state)
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
