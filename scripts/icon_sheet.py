"""Dev tool: PNG contact sheet of every tray icon state and frame (16/24/32 px).

Each frame is drawn magnified 8x (nearest neighbour) next to its true-size version, on a dark
and a light background. Labels use a tiny built-in 3x5 bitmap font (no external font).
Usage: uv run python scripts/icon_sheet.py [output.png]   (default agent-reports/tray-icons/sheet.png)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from parley import tray_icons  # noqa: E402

SIZES = (16, 24, 32)
MAG = 8
GAP = 12
BACKGROUNDS = (("dark", (0x20, 0x20, 0x20), (230, 230, 230)), ("light", (0xF3, 0xF3, 0xF3), (30, 30, 30)))

_FONT_ROWS = {
    "a": "010101111101101", "b": "110101110101110", "c": "011100100100011", "d": "110101101101110",
    "e": "111100110100111", "f": "111100110100100", "g": "011100101101011", "h": "101101111101101",
    "i": "111010010010111", "k": "101101110101101", "l": "100100100100111", "n": "110101101101101",
    "o": "010101101101010", "p": "110101110100100", "r": "110101110101101", "s": "011100010001110",
    "t": "111010010010010", "u": "101101101101111", "x": "101101010101101", "m": "101111111101101",
    "0": "111101101101111", "1": "010110010010111", "2": "110001010100111", "3": "110001010001110",
    "4": "101101111001001", "5": "111100110001110", "6": "011100111101111", "7": "111001010100100",
    "8": "111101111101111", "9": "111101111001110", "-": "000000111000000", " ": "0" * 15,
}


def draw_text(img: np.ndarray, x: int, y: int, text: str, colour, scale: int = 3) -> None:
    for ch in text:
        bits = _FONT_ROWS.get(ch, _FONT_ROWS[" "])
        for i, b in enumerate(bits):
            if b == "1":
                r, c = divmod(i, 3)
                img[y + r * scale : y + (r + 1) * scale, x + c * scale : x + (c + 1) * scale] = colour
        x += 4 * scale


def blend(rgba: np.ndarray, bg) -> np.ndarray:
    a = rgba[..., 3:4].astype(float) / 255
    return np.round(rgba[..., :3] * a + np.array(bg) * (1 - a)).astype(np.uint8)


def paste(img, x, y, tile):
    img[y : y + tile.shape[0], x : x + tile.shape[1]] = tile


def build() -> np.ndarray:
    max_frames = max(len(s.frames) for s in tray_icons.STATES.values())
    group_w = [max_frames * (s * MAG + s + GAP) + GAP for s in SIZES]
    label_w = 4 * 3 * 10 + 2 * GAP
    row_h = 32 * MAG + 2 * GAP
    width = label_w + sum(group_w)
    panel_h = len(tray_icons.STATE_NAMES) * row_h + 2 * GAP + 20
    sheet = np.zeros((panel_h * len(BACKGROUNDS), width, 3), np.uint8)
    for bi, (bname, bg, fg) in enumerate(BACKGROUNDS):
        top = bi * panel_h
        sheet[top : top + panel_h] = bg
        draw_text(sheet, GAP, top + 4, f"{bname} background - frames at 16 24 32 px, 8x and true size", fg, 2)
        for ri, state in enumerate(tray_icons.STATE_NAMES):
            y0 = top + 20 + GAP + ri * row_h
            draw_text(sheet, GAP, y0, state, fg, 3)
            x = label_w
            for si, size in enumerate(SIZES):
                for fi, frame in enumerate(tray_icons.frames(state, size)):
                    tile = blend(frame, bg)
                    big = tile.repeat(MAG, axis=0).repeat(MAG, axis=1)
                    paste(sheet, x, y0, big)
                    x += size * MAG + 4
                    paste(sheet, x, y0, tile)
                    x += size + GAP - 4
                x += (max_frames - len(tray_icons.frames(state, size))) * (size * MAG + size + GAP)
                x += GAP
            # sizes label under the first group
    return sheet


def main(argv: list[str]) -> int:
    out = Path(argv[1]) if len(argv) > 1 else ROOT / "agent-reports" / "tray-icons" / "sheet.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet = build()
    rgba = np.dstack([sheet, np.full(sheet.shape[:2], 255, np.uint8)])
    tray_icons.write_png(out, rgba)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
