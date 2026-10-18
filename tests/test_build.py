"""Build helpers (scripts/build.py) that need no PyInstaller: _build.py text, file index, path rule, install layout."""
import hashlib
import importlib.util
import struct
import sys
import tempfile
import unittest
from pathlib import Path

from parley import install, launcher, update

_spec = importlib.util.spec_from_file_location("parley_build_script", Path(__file__).resolve().parent.parent / "scripts" / "build.py")
build = importlib.util.module_from_spec(_spec)
sys.modules["parley_build_script"] = build
_spec.loader.exec_module(build)


def write_tree(root: Path, files: dict[str, bytes]) -> None:
    for rel, data in files.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_bytes(data)


GOOD = {"parley-app.exe": b"MZ" * 10, "parley-tray.exe": b"MZ" * 10, "_internal/python312.dll": b"d" * 3000, "_internal/pkg/data.json": b"{}"}


class BuildPy(unittest.TestCase):
    def test_default_is_updates_off_windows_x64(self):
        ns: dict = {}
        exec(build.build_py_text(""), ns)
        self.assertEqual((ns["FEED_URL"], ns["PLATFORM"], ns["PUBLIC_KEY"]), ("", "windows-x64", None))

    def test_feed_url_is_quoted_not_executed(self):
        ns: dict = {}
        exec(build.build_py_text('x"\nimport os'), ns)
        self.assertEqual(ns["FEED_URL"], 'x"\nimport os')


class Index(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="parley_test_build_")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_index_matches_installer_format_and_validates(self):
        app = self.root / "app"
        write_tree(app, GOOD)
        entries = build.scan_tree(app)
        self.assertEqual([e[0] for e in entries], sorted(GOOD))
        files = [update.IndexFile(p, h, n) for p, h, n in entries]
        self.assertEqual(build.index_bytes("1.2.3", entries), install._index_bytes("1.2.3", files))
        update.validate_index_files([{"path": p, "sha256": h, "size": n} for p, h, n in entries])
        sha = build.write_index(app, "1.2.3")
        raw = (app / launcher.INDEX_NAME).read_bytes()
        self.assertEqual(sha, hashlib.sha256(raw).hexdigest())
        self.assertEqual(update.parse_index(raw, sha, "1.2.3")[0].path, entries[0][0])
        launcher.verify_tree(app, launcher.read_index_entries(app, sha))  # the launcher accepts exactly this tree

    def test_index_excludes_itself_on_rescan(self):
        app = self.root / "app"
        write_tree(app, GOOD)
        build.write_index(app, "1.0.0")
        self.assertNotIn(launcher.INDEX_NAME, [e[0] for e in build.scan_tree(app)])

    def test_path_rule(self):
        e = lambda p: (p, "0" * 64, 1)  # noqa: E731
        ok = ["_internal/a-b_c.1/x.pyd", "parley-app.exe", "av.libs/libstdc++-6-x.dll"]
        bad = ["voices/!v/Alex", "a b.txt", "caf\u00e9.txt", "con.txt", "x/../y", "a/b./c"]
        self.assertEqual(build.bad_paths([e(p) for p in ok]), [])
        self.assertEqual(build.bad_paths([e(p) for p in bad]), bad)

    def test_rejected_paths_fail_the_build_unless_allowed(self):
        app = self.root / "app"
        write_tree(app, {**GOOD, "lib/c!.dll": b"x"})
        with self.assertRaises(build.BuildError):
            build.write_index(app, "1.0.0")
        self.assertFalse((app / launcher.INDEX_NAME).exists())
        build.write_index(app, "1.0.0", allow_rejected=True)
        self.assertTrue((app / launcher.INDEX_NAME).exists())


def fake_pe(subsystem: int) -> bytes:
    """Smallest thing pe_subsystem accepts: DOS header -> PE signature -> COFF header -> optional header with Subsystem."""
    dos = bytearray(64)
    dos[:2] = b"MZ"
    struct.pack_into("<I", dos, 0x3C, 64)
    opt = bytearray(96)
    struct.pack_into("<H", opt, 68, subsystem)
    return bytes(dos) + b"PE\0\0" + bytes(20) + bytes(opt)


class Executables(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="parley_test_build_")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def tree(self, launcher_sub=2, tray_sub=2, app_sub=3) -> Path:
        app = launcher.app_path(self.root, "1.0.0")
        app.mkdir(parents=True)
        (self.root / "Parley.exe").write_bytes(fake_pe(launcher_sub))
        (app / launcher.RUN_EXE).write_bytes(fake_pe(tray_sub))
        (app / launcher.CHILD_EXE).write_bytes(fake_pe(app_sub))
        return self.root

    def test_subsystem_is_read_from_the_pe_header(self):
        p = self.root / "a.exe"
        p.write_bytes(fake_pe(2))
        self.assertEqual(build.pe_subsystem(p), build.SUBSYSTEM_GUI)
        p.write_bytes(fake_pe(3))
        self.assertEqual(build.pe_subsystem(p), build.SUBSYSTEM_CONSOLE)
        p.write_bytes(b"not an exe at all" * 10)
        with self.assertRaises(build.BuildError):
            build.pe_subsystem(p)

    def test_windowed_launcher_and_tray_with_console_app_pass(self):
        build.check_exes(self.tree(), "1.0.0")

    def test_wrong_subsystem_fails_the_build(self):
        for kw in ({"launcher_sub": 3}, {"tray_sub": 3}, {"app_sub": 2}):
            with self.subTest(kw), tempfile.TemporaryDirectory(prefix="parley_test_build_") as d:
                self.root = Path(d)
                with self.assertRaises(build.BuildError):
                    build.check_exes(self.tree(**kw), "1.0.0")

    def test_missing_exe_fails_the_build(self):
        root = self.tree()
        (launcher.app_path(root, "1.0.0") / launcher.RUN_EXE).unlink()
        with self.assertRaises(build.BuildError):
            build.check_exes(root, "1.0.0")

    def test_icon_is_a_valid_ico_with_the_app_sizes(self):
        ico = build.write_icon(self.root / "x" / "parley.ico")
        data = ico.read_bytes()
        reserved, kind, count = struct.unpack_from("<HHH", data)
        self.assertEqual((reserved, kind, count), (0, 1, 5))
        for i in range(count):  # every image is a PNG inside the file
            size, offset = struct.unpack_from("<II", data, 6 + 16 * i + 8)
            self.assertEqual(data[offset:offset + 8], b"\x89PNG\r\n\x1a\n")
            self.assertLessEqual(offset + size, len(data))

    def test_spec_builds_two_exes_in_one_collect_and_a_windowed_launcher(self):
        spec = (Path(build.__file__).parent.parent / "packaging" / "parley.spec").read_text("utf-8")
        self.assertIn('name="parley-app", console=True', spec)
        self.assertIn('name="parley-tray", console=False', spec)
        self.assertIn('name="Parley", console=False', spec)
        self.assertIn("COLLECT(app_exe, tray_exe,", spec)
        self.assertEqual(spec.count("icon=ICON"), 3)
        self.assertIn(launcher.RUN_EXE, "parley-tray.exe")  # the exe name is what implies --tray
        from parley import tray_mode
        self.assertEqual(tray_mode.TRAY_EXE_STEM + ".exe", launcher.RUN_EXE)


class Layout(unittest.TestCase):
    def test_layout_is_what_the_launcher_expects(self):
        with tempfile.TemporaryDirectory(prefix="parley_test_build_") as t:
            t = Path(t)
            write_tree(t / "frozen_app", GOOD)
            (t / "Parley.exe").write_bytes(b"launcher")
            out = t / "install"
            appdir = build.layout(t / "frozen_app", t / "Parley.exe", out, "0.4.2")
            self.assertEqual(appdir, launcher.app_path(out, "0.4.2"))
            self.assertTrue((out / "Parley.exe").is_file())
            self.assertTrue((appdir / launcher.CHILD_EXE).is_file())
            ptr = launcher.read_pointer(out / launcher.POINTER)
            self.assertEqual((ptr.version, ptr.state, ptr.previous), ("0.4.2", "good", None))
            ptr.state = "pending"  # a pending start re-verifies the whole tree against the pointer's index hash
            self.assertTrue(launcher.tree_ok(out, ptr))
            self.assertEqual(launcher.resolve(out).version, "0.4.2")


if __name__ == "__main__":
    unittest.main()
