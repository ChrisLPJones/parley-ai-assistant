"""Tray icons drawn in code: a speech bubble whose colour AND glyph show Parley's state.

numpy only (no Pillow, no image assets). Shapes are signed-distance fields rendered at 4x
supersample and box-filtered down, so edges are antialiased at 16 px. Frames are rendered once
per (state, size) and cached. The same renderer writes `parley.ico` (PNG-in-ICO) and PNG files,
using only zlib/struct.
"""

from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

SUPERSAMPLE = 4
SIZES = (16, 20, 24, 32, 48)
ICO_SIZES = (16, 24, 32, 48, 256)

WHITE = (255, 255, 255)
SLATE = (0x64, 0x74, 0x8B)
RED = (0xEF, 0x44, 0x44)
BLUE = (0x3B, 0x82, 0xF6)
GREEN = (0x22, 0xC5, 0x5E)
GREY = (0x9C, 0xA3, 0xAF)
ORANGE = (0xF5, 0x9E, 0x0B)

# Element tuples: ("circle", cx, cy, r, strength) | ("box", cx, cy, hx, hy, radius, strength)
# | ("ring", cx, cy, r, half_width, strength). Unit square, y down. strength 1 = pure white.


@dataclass(frozen=True)
class StateSpec:
    colour: tuple[int, int, int]
    fps: float  # animation rate; irrelevant for single-frame states
    frames: tuple[tuple[float, tuple[tuple, ...]], ...]  # (alpha, glyph elements) per frame


_CX, _CY = 0.5, 0.42  # glyph centre = centre of the bubble body


def _dots(lit: int) -> tuple[tuple, ...]:
    return tuple(
        ("circle", x, _CY, 0.095 if i == lit else 0.065, 1.0 if i == lit else 0.45)
        for i, x in enumerate((0.27, 0.5, 0.73))
    )


def _bars(halves: tuple[float, float, float]) -> tuple[tuple, ...]:
    return tuple(("box", x, _CY, 0.055, h, 0.05, 1.0) for x, h in zip((0.29, 0.5, 0.71), halves))


STATES: dict[str, StateSpec] = {
    "idle": StateSpec(SLATE, 1.0, ((1.0, (("circle", _CX, _CY, 0.09, 1.0),)),)),
    "listening": StateSpec(
        RED, 2.0,
        ((1.0, (("circle", _CX, _CY, 0.18, 1.0),)), (1.0, (("circle", _CX, _CY, 0.11, 1.0),))),
    ),
    "thinking": StateSpec(BLUE, 4.0, tuple((1.0, _dots(i)) for i in range(3))),
    "speaking": StateSpec(
        GREEN, 6.0,
        (
            (1.0, _bars((0.10, 0.22, 0.15))),
            (1.0, _bars((0.20, 0.12, 0.22))),
            (1.0, _bars((0.14, 0.20, 0.09))),
        ),
    ),
    "starting": StateSpec(
        GREY, 2.0,
        ((0.35, (("ring", _CX, _CY, 0.145, 0.04, 1.0),)), (0.65, (("ring", _CX, _CY, 0.145, 0.04, 1.0),))),
    ),
    "error": StateSpec(
        ORANGE, 1.0,
        ((1.0, (("box", _CX, 0.33, 0.05, 0.12, 0.05, 1.0), ("circle", _CX, 0.585, 0.065, 1.0))),),
    ),
}
STATE_NAMES = tuple(STATES)


def _sd_rbox(px, py, cx, cy, hx, hy, r):
    qx = np.abs(px - cx) - (hx - r)
    qy = np.abs(py - cy) - (hy - r)
    outside = np.hypot(np.maximum(qx, 0), np.maximum(qy, 0))
    return outside + np.minimum(np.maximum(qx, qy), 0) - r


def _sd_segment(px, py, ax, ay, bx, by):
    dx, dy = bx - ax, by - ay
    t = np.clip(((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy), 0, 1)
    return np.hypot(px - (ax + t * dx), py - (ay + t * dy))


def _sd_triangle(px, py, a, b, c):
    d = np.minimum(
        np.minimum(_sd_segment(px, py, *a, *b), _sd_segment(px, py, *b, *c)),
        _sd_segment(px, py, *c, *a),
    )

    def side(p, q):
        return (q[0] - p[0]) * (py - p[1]) - (q[1] - p[1]) * (px - p[0])

    s1, s2, s3 = side(a, b), side(b, c), side(c, a)
    inside = ((s1 >= 0) & (s2 >= 0) & (s3 >= 0)) | ((s1 <= 0) & (s2 <= 0) & (s3 <= 0))
    return np.where(inside, -d, d)


def _sd_element(el, px, py):
    kind = el[0]
    if kind == "circle":
        _, cx, cy, r, _s = el
        return np.hypot(px - cx, py - cy) - r
    if kind == "box":
        _, cx, cy, hx, hy, rad, _s = el
        return _sd_rbox(px, py, cx, cy, hx, hy, rad)
    _, cx, cy, r, half, _s = el
    return np.abs(np.hypot(px - cx, py - cy) - r) - half


def _coverage(sd, scale):
    return np.clip(0.5 - sd * scale, 0.0, 1.0)


def _render_frame(size: int, colour, alpha: float, elements) -> np.ndarray:
    big = size * SUPERSAMPLE
    axis = (np.arange(big) + 0.5) / big
    px, py = np.meshgrid(axis, axis)
    body = np.minimum(
        _sd_rbox(px, py, 0.5, 0.42, 0.45, 0.33, 0.19),
        _sd_triangle(px, py, (0.13, 0.60), (0.45, 0.60), (0.11, 0.94)),
    )
    cov = _coverage(body, big)
    rgb = np.empty((big, big, 3))
    rgb[:] = colour
    for el in elements:
        m = (_coverage(_sd_element(el, px, py), big) * el[-1] * cov)[..., None]
        rgb = rgb * (1 - m) + np.array(WHITE) * m
    a = cov * alpha
    pre = np.dstack([rgb * a[..., None], a])  # premultiplied, so edges average correctly
    pre = pre.reshape(size, SUPERSAMPLE, size, SUPERSAMPLE, 4).mean(axis=(1, 3))
    out = np.zeros((size, size, 4), np.uint8)
    pa = pre[..., 3]
    nz = pa > 1e-6
    out[..., :3][nz] = np.clip(np.round(pre[..., :3][nz] / pa[nz][:, None]), 0, 255)
    out[..., 3] = np.clip(np.round(pa * 255), 0, 255)
    return out


@lru_cache(maxsize=None)
def frames(state: str, size: int) -> tuple[np.ndarray, ...]:
    """Straight (non-premultiplied) RGBA uint8 frames, shape (size, size, 4), rendered once."""
    spec = STATES[state]
    out = []
    for alpha, elements in spec.frames:
        arr = _render_frame(size, spec.colour, alpha, elements)
        arr.setflags(write=False)
        out.append(arr)
    return tuple(out)


def fps(state: str) -> float:
    return STATES[state].fps


# ---- file writers (zlib/struct only) ----------------------------------------------------------

def png_bytes(rgba: np.ndarray) -> bytes:
    h, w, _ = rgba.shape
    raw = b"".join(b"\x00" + rgba[y].tobytes() for y in range(h))

    def chunk(tag: bytes, data: bytes) -> bytes:
        body = tag + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)

    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, 9))
        + chunk(b"IEND", b"")
    )


def write_png(path: str | Path, rgba: np.ndarray) -> None:
    Path(path).write_bytes(png_bytes(rgba))


def write_pngs(directory: str | Path, state: str = "idle", sizes=SIZES) -> list[Path]:
    """Write <state>-<size>[-<frame>].png files; returns the paths."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for size in sizes:
        for i, frame in enumerate(frames(state, size)):
            p = directory / f"{state}-{size}-{i}.png"
            write_png(p, frame)
            paths.append(p)
    return paths


def ico_bytes(state: str = "idle", sizes=ICO_SIZES) -> bytes:
    """PNG-in-ICO (Vista+), first frame of `state` at each size."""
    images = [png_bytes(frames(state, s)[0]) for s in sizes]
    head = struct.pack("<HHH", 0, 1, len(images))
    offset = 6 + 16 * len(images)
    entries = b""
    for s, data in zip(sizes, images):
        entries += struct.pack("<BBBBHHII", 0 if s >= 256 else s, 0 if s >= 256 else s, 0, 0, 1, 32, len(data), offset)
        offset += len(data)
    return head + entries + b"".join(images)


def write_ico(path: str | Path, state: str = "idle", sizes=ICO_SIZES) -> None:
    Path(path).write_bytes(ico_bytes(state, sizes))
