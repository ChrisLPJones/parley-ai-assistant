"""Throwaway demo: `python -m parley.tray_demo [--seconds N]` shows the REAL tray icon cycling
through every state every ~2 s, with a menu wired to harmless actions (Quit exits).
No Parley models or app code are loaded."""

from __future__ import annotations

import argparse
import logging
import threading

from . import tray_icons
from .tray import Tray, default_items

log = logging.getLogger("parley.tray_demo")
STEP_S = 2.0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="parley.tray_demo")
    ap.add_argument("--seconds", type=float, default=0, help="exit by itself after N seconds (0 = until Quit)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    tray = Tray(on_activate=lambda: tray.balloon("Parley demo", "Left click: Settings would open here."))
    state = {"name": "starting"}

    def menu() -> None:
        tray.set_menu(default_items(
            f"State: {state['name']}",
            on_settings=lambda: tray.balloon("Parley demo", "Settings would open here."),
            on_check_updates=lambda: tray.balloon("Parley demo", "You are up to date (demo)."),
            on_open_logs=lambda: log.info("Open log folder (demo, does nothing)"),
            on_quit=tray.stop,
        ))

    stop = threading.Event()

    def cycle() -> None:
        tray.wait_ready()
        tray.balloon("Parley demo", "Cycling through every icon state. Right-click for the menu.")
        for name in tray_icons.STATE_NAMES * 1000:
            if stop.is_set():
                return
            state["name"] = name
            tray.set_state(name)
            tray.set_tooltip(f"Parley demo: {name}")
            menu()
            log.info("state: %s", name)
            stop.wait(STEP_S)

    menu()
    threading.Thread(target=cycle, daemon=True).start()
    timer = None
    if args.seconds:
        timer = threading.Timer(args.seconds, tray.stop)
        timer.daemon = True
        timer.start()
    try:
        tray.run()
    except KeyboardInterrupt:
        pass
    finally:
        if timer is not None:
            timer.cancel()
        stop.set()
        tray.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
