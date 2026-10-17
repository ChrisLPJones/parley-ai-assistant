import argparse
import logging
import os
import sys
import time
from pathlib import Path

from . import __version__, config


def _parse_key(name: str):
    from pynput import keyboard

    name = name.strip().lower()
    if hasattr(keyboard.Key, name):
        return getattr(keyboard.Key, name)
    if len(name) == 1:
        return keyboard.KeyCode.from_char(name)
    raise SystemExit(f"Unknown hotkey {name!r}. Use a pynput key name like pause, f13, scroll_lock, ctrl_r.")


def run_hotkey_loop(app, key_name: str) -> None:
    from pynput import keyboard

    key = _parse_key(key_name)

    def on_press(k):
        if k == key:
            app.begin_listening()

    def on_release(k):
        if k == key:
            app.end_listening()

    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        logging.info("Ready. Hold [%s] and speak; release to send. Ctrl+C to quit.", key_name)
        while listener.is_alive():
            listener.join(0.5)  # short joins keep Ctrl+C responsive on Windows


def main() -> None:
    parser = argparse.ArgumentParser(prog="parley", description="Local push-to-talk voice assistant.")
    parser.add_argument("--config", type=Path, help=f"config file (default: {config.default_config_path()})")
    parser.add_argument("--ask", metavar="TEXT", help="ask one typed question, speak the answer, and exit")
    parser.add_argument("--list-devices", action="store_true", help="list audio devices and exit")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--version", action="version", version=f"parley {__version__}")
    args = parser.parse_args()
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(message)s", datefmt="%H:%M:%S",
    )
    # Libraries like huggingface_hub re-enable httpx request logging, so filter at the handler instead
    noisy = ("httpx", "httpcore", "huggingface_hub", "faster_whisper", "phonemizer")
    for handler in logging.getLogger().handlers:
        handler.addFilter(lambda r: r.levelno >= logging.ERROR or not r.name.startswith(noisy))

    if args.list_devices:
        import sounddevice as sd
        print(sd.query_devices())
        return

    cfg = config.load(args.config)
    from .app import Parley

    app = Parley(cfg, use_mic=args.ask is None)
    try:
        if args.ask:
            app.ask(args.ask)
            while not app.finished.wait(0.5):  # short waits keep Ctrl+C responsive on Windows
                pass
            time.sleep(0.3)  # let the output buffer drain
        else:
            run_hotkey_loop(app, cfg.hotkey.key)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    sys.exit(main())
