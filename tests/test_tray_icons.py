"""Tests for the tray icon renderer: shapes, frames, distinctness, antialiasing, PNG and ICO output."""

import itertools
import struct
import tempfile
import unittest
import zlib
from pathlib import Path

import numpy as np

from parley import tray_icons as ti


def parse_png(data: bytes) -> np.ndarray:
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    pos, idat, ihdr = 8, b"", None
    while pos < len(data):
        (n,) = struct.unpack(">I", data[pos : pos + 4])
        tag, body = data[pos + 4 : pos + 8], data[pos + 8 : pos + 8 + n]
        (crc,) = struct.unpack(">I", data[pos + 8 + n : pos + 12 + n])
        assert crc == zlib.crc32(tag + body) & 0xFFFFFFFF
        if tag == b"IHDR":
            ihdr = struct.unpack(">IIBBBBB", body)
        elif tag == b"IDAT":
            idat += body
        pos += 12 + n
    w, h, depth, ctype, *_ = ihdr
    assert (depth, ctype) == (8, 6)
    raw = zlib.decompress(idat)
    stride = 1 + w * 4
    rows = []
    for y in range(h):
        assert raw[y * stride] == 0
        rows.append(np.frombuffer(raw[y * stride + 1 : (y + 1) * stride], np.uint8).reshape(w, 4))
    return np.array(rows)


class RendererTests(unittest.TestCase):
    EXPECTED_FRAMES = {"idle": 1, "listening": 2, "thinking": 3, "speaking": 3, "starting": 2, "error": 1}

    def test_frame_counts_shapes_dtypes(self):
        self.assertEqual({s: len(v.frames) for s, v in ti.STATES.items()}, self.EXPECTED_FRAMES)
        for state, n in self.EXPECTED_FRAMES.items():
            for size in ti.SIZES:
                fr = ti.frames(state, size)
                self.assertEqual(len(fr), n)
                for f in fr:
                    self.assertEqual(f.shape, (size, size, 4))
                    self.assertEqual(f.dtype, np.uint8)

    def test_frames_are_cached(self):
        self.assertIs(ti.frames("idle", 16), ti.frames("idle", 16))

    def test_colour_constants(self):
        self.assertEqual(ti.SLATE, (0x64, 0x74, 0x8B))
        self.assertEqual(ti.RED, (0xEF, 0x44, 0x44))
        self.assertEqual(ti.BLUE, (0x3B, 0x82, 0xF6))
        self.assertEqual(ti.GREEN, (0x22, 0xC5, 0x5E))
        self.assertEqual(ti.ORANGE, (0xF5, 0x9E, 0x0B))
        for state, rgb in (("idle", ti.SLATE), ("listening", ti.RED), ("thinking", ti.BLUE),
                           ("speaking", ti.GREEN), ("error", ti.ORANGE)):
            self.assertEqual(ti.STATES[state].colour, rgb)
        # a body pixel left of the glyph carries the state colour, fully opaque
        f = ti.frames("idle", 48)[0]
        self.assertEqual(tuple(f[20, 8, :3]), ti.SLATE)
        self.assertEqual(f[20, 8, 3], 255)

    def test_starting_is_translucent(self):
        for f in ti.frames("starting", 32):
            self.assertTrue(0.25 < f[..., 3].max() / 255 < 0.8)

    def test_states_pairwise_distinct_at_every_size(self):
        for size in ti.SIZES:
            for a, b in itertools.combinations(ti.STATES, 2):
                same = any(np.array_equal(x, y) for x in ti.frames(a, size) for y in ti.frames(b, size))
                self.assertFalse(same, f"{a} and {b} share a frame at {size}px")

    def test_glyph_differs_not_only_colour(self):
        for size in ti.SIZES:
            sigs = {}
            for state in ti.STATES:
                f = ti.frames(state, size)[0]
                sigs[state] = (f[..., :3].min(axis=2) > 200) & (f[..., 3] > 100)  # whitish pixels
            for a, b in itertools.combinations(sigs, 2):
                self.assertFalse(np.array_equal(sigs[a], sigs[b]), f"{a}/{b} glyph identical at {size}")

    def test_frames_within_a_state_differ(self):
        for state in ti.STATES:
            for x, y in itertools.combinations(ti.frames(state, 32), 2):
                self.assertFalse(np.array_equal(x, y), state)

    def test_edges_antialiased(self):
        for size in ti.SIZES:
            a = ti.frames("idle", size)[0][..., 3]
            self.assertGreater(((a > 0) & (a < 255)).sum(), size)  # a real fringe, not a 1-bit mask
            self.assertEqual(a[0, 0], 0)
            self.assertEqual(a[size // 2 - 2, size // 2 - 2], 255)

    def test_fps_defined(self):
        for s in ti.STATES:
            self.assertGreater(ti.fps(s), 0)


class FileTests(unittest.TestCase):
    def test_png_roundtrip(self):
        f = ti.frames("thinking", 24)[1]
        np.testing.assert_array_equal(parse_png(ti.png_bytes(f)), f)

    def test_ico_valid(self):
        data = ti.ico_bytes()
        self.assertEqual(struct.unpack("<HHH", data[:6]), (0, 1, len(ti.ICO_SIZES)))
        end = 0
        for i, size in enumerate(ti.ICO_SIZES):
            w, h, _c, _r, planes, bpp, length, offset = struct.unpack("<BBBBHHII", data[6 + 16 * i : 22 + 16 * i])
            self.assertEqual((w, h), (0, 0) if size == 256 else (size, size))
            self.assertEqual((planes, bpp), (1, 32))
            self.assertEqual(parse_png(data[offset : offset + length]).shape, (size, size, 4))
            end = offset + length
        self.assertEqual(end, len(data))

    def test_write_files(self):
        with tempfile.TemporaryDirectory(prefix="parley_test_tray_") as d:
            ti.write_ico(Path(d) / "parley.ico")
            self.assertGreater((Path(d) / "parley.ico").stat().st_size, 100)
            paths = ti.write_pngs(d, "listening", sizes=(16,))
            self.assertEqual(len(paths), 2)
            parse_png(paths[0].read_bytes())


if __name__ == "__main__":
    unittest.main()
