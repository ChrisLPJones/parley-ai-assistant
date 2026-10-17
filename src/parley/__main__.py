import argparse
import logging
import logging.handlers
import os
import queue
import sys
import threading
import time
from pathlib import Path

from . import __version__, config

log = logging.getLogger(__name__)


def _parse_key(name: str):
    from pynput import keyboard

    name = name.strip().lower()
    if hasattr(keyboard.Key, name):
        return getattr(keyboard.Key, name)
    if len(name) == 1:
        return keyboard.KeyCode.from_char(name)
    raise SystemExit(f"Unknown hotkey {name!r}. Use a pynput key name like pause, f13, scroll_lock, ctrl_r.")


def run_hotkey_loop(app, key_name: str, on_ready=None, stop: threading.Event | None = None) -> None:
    from pynput import keyboard

    key = _parse_key(key_name)
    vk = getattr(getattr(key, "value", key), "vk", None)
    listener = None

    def on_press(k):
        if k == key:
            app.begin_listening()

    def on_release(k):
        if k == key:
            app.end_listening()

    def win32_filter(msg, data):
        # Swallow the hotkey so no other app reacts to it. Without this, Pause in a focused
        # console window freezes that console until another key is pressed.
        if vk is None or data.vkCode != vk:
            return True
        if msg in (0x100, 0x104):  # WM_KEYDOWN, WM_SYSKEYDOWN
            app.begin_listening()
        elif msg in (0x101, 0x105):  # WM_KEYUP, WM_SYSKEYUP
            app.end_listening()
        listener.suppress_event()

    listener = keyboard.Listener(on_press=on_press, on_release=on_release, win32_event_filter=win32_filter)
    with listener:
        logging.info("Ready. Hold [%s] and speak; release to send. Ctrl+C to quit.", key_name)
        if on_ready:
            on_ready()
        while listener.is_alive() and not (stop and stop.is_set()):
            listener.join(0.5)  # short joins keep Ctrl+C responsive on Windows


def _disable_console_quick_edit() -> None:
    """Clicking in a Windows console normally starts a text selection that freezes the program
    writing to it; turn that off for Parley's console."""
    if sys.platform != "win32":
        return
    import ctypes

    k32 = ctypes.windll.kernel32
    handle = k32.GetStdHandle(-10)  # STD_INPUT_HANDLE
    mode = ctypes.c_uint32()
    if k32.GetConsoleMode(handle, ctypes.byref(mode)):
        k32.SetConsoleMode(handle, (mode.value & ~0x0040) | 0x0080)  # -QUICK_EDIT, +EXTENDED_FLAGS


def self_check(config_path: Path | None = None) -> int:
    """Smoke test for a freshly built copy: config loads, every parley module imports, the hotkey, audio and model
    libraries import. Opens no microphone and makes no network request. Returns the exit code (0 = fine)."""
    import importlib
    import pkgutil

    log = logging.getLogger("parley.self_check")
    problems: list[str] = []

    def attempt(what: str, fn) -> None:
        try:
            fn()
        except BaseException as e:  # a missing DLL can surface as anything, incl. SystemExit from a config error
            problems.append(what)
            log.error("self-check: %s failed: %s: %s", what, type(e).__name__, e)

    attempt("config", lambda: config.load(config_path))
    for mod in pkgutil.iter_modules(__import__("parley").__path__):
        if mod.name != "__main__":
            attempt(f"module parley.{mod.name}", lambda n=mod.name: importlib.import_module(f"parley.{n}"))
    for lib in ("pynput.keyboard", "sounddevice", "numpy", "httpx", "faster_whisper", "ctranslate2",
                "onnxruntime", "kokoro_onnx"):
        attempt(f"library {lib}", lambda n=lib: importlib.import_module(n))
    if problems:
        print(f"parley {__version__} self-check FAILED: {', '.join(problems)}", file=sys.stderr)
        return 1
    print(f"parley {__version__} self-check ok")
    return 0


def test_mic(cfg, seconds: int = 4) -> None:
    import numpy as np
    import sounddevice as sd

    from .audio import MIC_RATE, _device
    from .config import app_dir
    from .stt import Transcriber

    device = _device(cfg.audio.input_device, "input")
    used = device if device is not None else sd.default.device[0]
    mme = next(i for i, h in enumerate(sd.query_hostapis()) if h["name"] == "MME") if sys.platform == "win32" else None
    print("Microphones (set [audio] input_device in config.toml to a name or number):")
    for i, dev in enumerate(sd.query_devices()):
        if dev["max_input_channels"] > 0 and (mme is None or dev["hostapi"] == mme):
            print(f"  {'>' if i == used else ' '} {i:3}  {dev['name']}")
    print(f"\nUsing: {sd.query_devices(used)['name']}")
    stt = Transcriber(cfg.stt, app_dir() / "models" / "whisper")
    input(f"Press Enter, then speak for {seconds} seconds...")
    audio = sd.rec(seconds * MIC_RATE, samplerate=MIC_RATE, channels=1, dtype="float32", device=device)
    for left in range(seconds, 0, -1):
        print(f"  recording... {left}", end="\r", flush=True)
        sd.sleep(1000)
    sd.wait()
    audio = audio[:, 0]
    peak = float(np.abs(audio).max())
    print(f"\nPeak level: {peak:.0%}" + ("  <- silent: this mic isn't picking anything up" if peak < 0.005 else "  (quiet)" if peak < 0.05 else ""))
    print(f"Heard: {stt.transcribe(audio) or '(nothing recognised)'}")


def create_update_flow(app, cfg):
    """The update flow, or None: unpackaged runs have none, and any failure creating it (e.g. a bad build-time feed URL)
    only turns updates off; it must never stop Parley from starting."""
    from . import autoupdate
    try:
        return autoupdate.create(app, cfg)
    except Exception as e:
        log.warning("Updates are off: %s", e)
        return None


def main() -> None:
    parser = argparse.ArgumentParser(prog="parley", description="Local push-to-talk voice assistant.")
    parser.add_argument("--config", type=Path, help=f"config file (default: {config.default_config_path()})")
    parser.add_argument("--ask", metavar="TEXT", help="ask one typed question, speak the answer, and exit")
    parser.add_argument("--list-devices", action="store_true", help="list audio devices and exit")
    parser.add_argument("--test-mic", action="store_true", help="record a few seconds and show the level and what was heard")
    parser.add_argument("--self-check", action="store_true",
                        help="check that the config loads and all modules and libraries import (no mic, no network), then exit")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--version", action="version", version=f"parley {__version__}")
    args = parser.parse_args()
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    if args.verbose:
        logging.getLogger("parley").setLevel(logging.DEBUG)  # Parley's own timings, not every library's
    # Libraries like huggingface_hub re-enable httpx request logging, so filter at the handler instead
    noisy = ("httpx", "httpcore", "huggingface_hub", "faster_whisper", "phonemizer", "ddgs", "primp")
    root = logging.getLogger()
    for handler in root.handlers:
        handler.addFilter(lambda r: r.levelno >= logging.ERROR or not r.name.startswith(noisy))
    # Write log lines from a background thread: if the console stops accepting output (text
    # selected, Pause pressed), only logging waits, never listening, thinking or speaking
    log_queue: queue.Queue = queue.Queue()
    logging.handlers.QueueListener(log_queue, *root.handlers, respect_handler_level=True).start()
    root.handlers = [logging.handlers.QueueHandler(log_queue)]
    _disable_console_quick_edit()

    if args.self_check:
        return self_check(args.config)
    if args.list_devices:
        import sounddevice as sd
        print(sd.query_devices())
        return

    cfg = config.load(args.config)
    if args.test_mic:
        test_mic(cfg)
        return
    from .app import Parley

    app = Parley(cfg, use_mic=args.ask is None)
    flow = None
    if args.ask is None:
        flow = create_update_flow(app, cfg)
    try:
        if args.ask:
            app.ask(args.ask)
            while not app.finished.wait(0.5):  # short waits keep Ctrl+C responsive on Windows
                pass
            time.sleep(0.3)  # let the output buffer drain
        else:
            run_hotkey_loop(app, cfg.hotkey.key, flow.start if flow else None, flow.restart_requested if flow else None)
    except KeyboardInterrupt:
        pass
    finally:
        if flow:
            flow.stop()
    if flow and flow.restart_requested.is_set():
        flow.restart()  # an update was installed: the launcher starts the new version


if __name__ == "__main__":
    sys.exit(main())
