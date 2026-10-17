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


def main() -> None:
    parser = argparse.ArgumentParser(prog="parley", description="Local push-to-talk voice assistant.")
    parser.add_argument("--config", type=Path, help=f"config file (default: {config.default_config_path()})")
    parser.add_argument("--ask", metavar="TEXT", help="ask one typed question, speak the answer, and exit")
    parser.add_argument("--list-devices", action="store_true", help="list audio devices and exit")
    parser.add_argument("--test-mic", action="store_true", help="record a few seconds and show the level and what was heard")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--version", action="version", version=f"parley {__version__}")
    args = parser.parse_args()
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    if args.verbose:
        logging.getLogger("parley").setLevel(logging.DEBUG)  # Parley's own timings, not every library's
    # Libraries like huggingface_hub re-enable httpx request logging, so filter at the handler instead
    noisy = ("httpx", "httpcore", "huggingface_hub", "faster_whisper", "phonemizer", "ddgs", "primp")
    for handler in logging.getLogger().handlers:
        handler.addFilter(lambda r: r.levelno >= logging.ERROR or not r.name.startswith(noisy))

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
