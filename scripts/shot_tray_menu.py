"""Dev tool: screenshots of the tray's right-click popup menu, dark and light (BitBlt of the screen around the menu).

    uv run python scripts/shot_tray_menu.py [--out DIR]

Opens the REAL popup from a throwaway Tray (no Parley, no models) at a fixed screen point by calling the same code the
right click runs, and captures it. The Windows theme is NOT changed: the "uses light theme" answer is patched in this
process only. Moves no mouse and presses no key (a Down key message is posted to the menu to show the highlight).
Needs a Windows desktop session.
"""
import argparse
import ctypes
import sys
import threading
import time
from ctypes import wintypes
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

POINT = (500, 300)
SIZE = (520, 420)


def grab_screen(x: int, y: int, w: int, h: int) -> bytes:
    user32, gdi32 = ctypes.windll.user32, ctypes.windll.gdi32
    for f, r in ((user32.GetDC, ctypes.c_void_p), (gdi32.CreateCompatibleDC, ctypes.c_void_p),
                 (gdi32.CreateCompatibleBitmap, ctypes.c_void_p), (gdi32.SelectObject, ctypes.c_void_p)):
        f.restype = r
    user32.GetDC.argtypes = [ctypes.c_void_p]
    gdi32.CreateCompatibleDC.argtypes = [ctypes.c_void_p]
    gdi32.CreateCompatibleBitmap.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
    gdi32.SelectObject.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    gdi32.BitBlt.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p,
                             ctypes.c_int, ctypes.c_int, ctypes.c_uint32]
    gdi32.GetDIBits.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint, ctypes.c_uint, ctypes.c_void_p,
                                ctypes.c_void_p, ctypes.c_uint]
    screen = user32.GetDC(None)
    mem = gdi32.CreateCompatibleDC(screen)
    bmp = gdi32.CreateCompatibleBitmap(screen, w, h)
    gdi32.SelectObject(mem, bmp)
    gdi32.BitBlt(mem, 0, 0, w, h, screen, x, y, 0x00CC0020 | 0x40000000)  # SRCCOPY | CAPTUREBLT (layered windows, menus)
    from shot_window import BITMAPINFOHEADER

    info = BITMAPINFOHEADER(ctypes.sizeof(BITMAPINFOHEADER), w, -h, 1, 32, 0)
    buf = ctypes.create_string_buffer(w * h * 4)
    gdi32.GetDIBits(mem, bmp, 0, h, buf, ctypes.byref(info), 0)
    raw = buf.raw
    rgb = bytearray(w * h * 3)
    rgb[0::3], rgb[1::3], rgb[2::3] = raw[2::4], raw[1::4], raw[0::4]
    return bytes(rgb)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out", type=Path, default=ROOT / "agent-reports" / "settings-window" / "shots")
    args = p.parse_args()
    from shot_window import png_bytes

    from parley import tray, ui

    ui.enable_dpi_awareness()
    import win32con
    import win32gui

    args.out.mkdir(parents=True, exist_ok=True)
    noop = lambda: None  # noqa: E731
    t = tray.Tray(tooltip="Parley menu shot")
    t.set_menu(tray.default_items("Ready - hold Pause to talk", on_settings=noop, on_check_updates=noop,
                                  on_open_logs=noop, on_quit=noop))
    th = threading.Thread(target=t.run, daemon=True)
    th.start()
    if not t.wait_ready(10) or not t.created:
        print("the tray window could not be created", file=sys.stderr)
        return 1
    time.sleep(0.5)
    real_cursor, real_theme = win32gui.GetCursorPos, ui.system_uses_light_theme
    win32gui.GetCursorPos = lambda: POINT
    try:
        for name in ("dark", "light"):
            ui.system_uses_light_theme = lambda n=name: n == "light"  # this process only: the Windows setting is untouched
            t._post(t._show_menu)  # what a right click runs; blocks the tray thread in the menu loop
            time.sleep(1.0)
            win32gui.PostMessage(t._hwnd, win32con.WM_KEYDOWN, win32con.VK_DOWN, 0)  # highlight the first enabled item
            time.sleep(0.2)
            win32gui.PostMessage(t._hwnd, win32con.WM_KEYDOWN, win32con.VK_DOWN, 0)
            time.sleep(0.6)
            rgb = grab_screen(POINT[0] - 10, POINT[1] - 10, *SIZE)
            path = args.out / f"tray-menu-{name}.png"
            path.write_bytes(png_bytes(SIZE[0], SIZE[1], rgb))
            print(path)
            t._post(ctypes.windll.user32.EndMenu)
            time.sleep(0.6)
    finally:
        win32gui.GetCursorPos, ui.system_uses_light_theme = real_cursor, real_theme
        t.stop()
        th.join(5)
    return 0


if __name__ == "__main__":
    sys.exit(main())
