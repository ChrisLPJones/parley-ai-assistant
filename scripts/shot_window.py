"""Dev tool: screenshots of the real Settings window (stub app, throwaway home) for light/dark x 100/150 %, General and Voice.

    uv run python scripts/shot_window.py [--out DIR] [--themes light dark] [--scales 100 150] [--no-sample]

Needs a Windows desktop session (PrintWindow of a live window). Writes PNGs to agent-reports/settings-window/shots/ by
default. `--no-sample` skips the extra 'sample playing' / 'restart needed' states (the stub never makes sound either way).
Nothing here touches the real Parley home: ui_preview forces PARLEY_HOME to a parley_test_* temp folder.
"""
import argparse
import ctypes
import struct
import sys
import time
import zlib
from ctypes import wintypes
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

PW_RENDERFULLCONTENT = 2


def png_bytes(width: int, height: int, rgb: bytes) -> bytes:
    """A truecolour PNG from packed RGB rows (zlib, no Pillow)."""
    raw = b"".join(b"\0" + rgb[y * width * 3:(y + 1) * width * 3] for y in range(height))

    def chunk(kind: bytes, data: bytes) -> bytes:
        body = kind + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)

    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))


class BITMAPINFOHEADER(ctypes.Structure):
    _fields_ = [("biSize", wintypes.DWORD), ("biWidth", wintypes.LONG), ("biHeight", wintypes.LONG),
                ("biPlanes", wintypes.WORD), ("biBitCount", wintypes.WORD), ("biCompression", wintypes.DWORD),
                ("biSizeImage", wintypes.DWORD), ("biXPelsPerMeter", wintypes.LONG), ("biYPelsPerMeter", wintypes.LONG),
                ("biClrUsed", wintypes.DWORD), ("biClrImportant", wintypes.DWORD)]


def grab(hwnd: int) -> tuple[int, int, bytes]:
    """(width, height, RGB bytes) of the window's visible frame, via PrintWindow."""
    user32, gdi32, dwm = ctypes.windll.user32, ctypes.windll.gdi32, ctypes.windll.dwmapi
    for fn in (user32.GetDC, user32.GetWindowRect, user32.PrintWindow, gdi32.CreateCompatibleDC,
               gdi32.CreateCompatibleBitmap, gdi32.SelectObject, gdi32.GetDIBits):
        fn.restype = ctypes.c_void_p if fn in (user32.GetDC, gdi32.CreateCompatibleDC, gdi32.CreateCompatibleBitmap,
                                               gdi32.SelectObject) else ctypes.c_int
    user32.GetDC.argtypes = [ctypes.c_void_p]
    user32.PrintWindow.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint]
    gdi32.CreateCompatibleDC.argtypes = [ctypes.c_void_p]
    gdi32.CreateCompatibleBitmap.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
    gdi32.SelectObject.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    gdi32.GetDIBits.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint, ctypes.c_uint, ctypes.c_void_p,
                                ctypes.c_void_p, ctypes.c_uint]
    gdi32.DeleteObject.argtypes = [ctypes.c_void_p]
    gdi32.DeleteDC.argtypes = [ctypes.c_void_p]
    user32.ReleaseDC.argtypes = [ctypes.c_void_p, ctypes.c_void_p]

    outer, frame = wintypes.RECT(), wintypes.RECT()
    user32.GetWindowRect(hwnd, ctypes.byref(outer))
    if dwm.DwmGetWindowAttribute(hwnd, 9, ctypes.byref(frame), ctypes.sizeof(frame)) != 0:  # EXTENDED_FRAME_BOUNDS
        frame = outer
    w, h = outer.right - outer.left, outer.bottom - outer.top
    screen = user32.GetDC(None)
    mem = gdi32.CreateCompatibleDC(screen)
    bmp = gdi32.CreateCompatibleBitmap(screen, w, h)
    old = gdi32.SelectObject(mem, bmp)
    try:
        if not user32.PrintWindow(hwnd, mem, PW_RENDERFULLCONTENT):
            raise RuntimeError("PrintWindow failed")
        info = BITMAPINFOHEADER(ctypes.sizeof(BITMAPINFOHEADER), w, -h, 1, 32, 0)  # top-down BGRA
        buf = ctypes.create_string_buffer(w * h * 4)
        if not gdi32.GetDIBits(mem, bmp, 0, h, buf, ctypes.byref(info), 0):
            raise RuntimeError("GetDIBits failed")
    finally:
        gdi32.SelectObject(mem, old)
        gdi32.DeleteObject(bmp)
        gdi32.DeleteDC(mem)
        user32.ReleaseDC(None, screen)
    left, top = frame.left - outer.left, frame.top - outer.top
    cw, ch = frame.right - frame.left, frame.bottom - frame.top
    data = buf.raw
    rows = []
    for y in range(top, top + ch):
        row = data[(y * w + left) * 4:(y * w + left + cw) * 4]
        rgb = bytearray(cw * 3)
        rgb[0::3], rgb[1::3], rgb[2::3] = row[2::4], row[1::4], row[0::4]  # BGRA -> RGB
        rows.append(bytes(rgb))
    return cw, ch, b"".join(rows)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out", type=Path, default=ROOT / "agent-reports" / "settings-window" / "shots")
    p.add_argument("--themes", nargs="+", choices=("light", "dark"), default=["light", "dark"])
    p.add_argument("--scales", nargs="+", type=int, default=[100, 150])
    p.add_argument("--no-sample", action="store_true", help="only the plain pages (skip the sample-playing and restart-needed states)")
    args = p.parse_args(argv)

    from parley import settings_window, ui, ui_preview

    args.out.mkdir(parents=True, exist_ok=True)
    host, home, created = ui_preview.build_host()
    written: list[Path] = []

    def current():
        return settings_window._current

    def shoot(name: str, page: str, theme: str, scale: int, state: str | None = None, act=None) -> None:
        ui.theme_override, ui.scale_override = theme, scale
        h, _, _ = ui_preview.build_host(state, home=home)
        settings_window.open_settings(h, page)
        end = time.monotonic() + 10
        while not ui.call(lambda: current() is not None and current().alive()):
            if time.monotonic() > end:
                raise SystemExit("the window did not open")
            time.sleep(0.05)
        if act:
            ui.call(lambda: act(current()))
        time.sleep(0.8)  # let Tk finish drawing and the DWM animation settle
        hwnd = ui.call(lambda: ctypes.windll.user32.GetParent(current().top.winfo_id()) or current().top.winfo_id())
        w, hgt, rgb = grab(hwnd)
        path = args.out / f"{name}-{theme}-{scale}.png"
        path.write_bytes(png_bytes(w, hgt, rgb))
        written.append(path)
        print(f"{path}  {w}x{hgt}")
        settings_window.close_settings()
        while ui.call(lambda: current() is not None and current().alive()):
            time.sleep(0.05)

    try:
        for theme in args.themes:
            for scale in args.scales:
                shoot("general", "general", theme, scale)
                shoot("voice", "voice", theme, scale)
                if not args.no_sample:
                    shoot("voice-playing", "voice", theme, scale, act=lambda w: w.sample_btn.invoke())
                    shoot("general-restart-needed", "general", theme, scale, state="restart-needed")
                    shoot("general-error", "general", theme, scale, state="error")
    finally:
        ui.stop()
        if created:
            import shutil

            shutil.rmtree(home, ignore_errors=True)
    print(f"{len(written)} screenshot(s) in {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
