"""install.py + launcher.py on fake app trees (stdlib unittest; temp dirs only, never the Parley home).

The fake child process is the Python executable running a tiny main.py from the app dir.
Run: uv run python -m unittest discover -s tests -v
"""
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import httpx

from parley import install, launcher, update
from parley.download import DownloadError
from parley.install import InstallError
from parley.update import UpdateError, Updater
from test_update import FakeServer, make_release, sha

CRLF = bytes([13, 10])
OK = b"import sys\nsys.exit(0)\n"
CRASH = b"import sys\nsys.exit(1)\n"


def child(appdir: Path, args: list[str]) -> list[str]:
    return [sys.executable, str(appdir / "main.py"), *args]


def snapshot(root: Path) -> dict[str, bytes]:
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()}


def tamper(path: Path, data: bytes) -> None:
    os.chmod(path, 0o666)
    path.write_bytes(data)


def tree(main: bytes = OK, **extra: bytes) -> dict[str, bytes]:
    return {"main.py": main, "data/same.bin": b"s" * 5000, "data/other.txt": b"other", **extra}


def seed(root: Path, version: str, files: dict[str, bytes]) -> None:
    """Make `version` the installed, good version."""
    d = launcher.app_path(root, version)
    entries = []
    for p, data in files.items():
        (d / p).parent.mkdir(parents=True, exist_ok=True)
        (d / p).write_bytes(data)
        entries.append(install.IndexFile(p, sha(data), len(data)))
    idx = install._index_bytes(version, entries)
    (d / launcher.INDEX_NAME).write_bytes(idx)
    launcher.write_pointer(root, launcher.Pointer(version, index_sha256=hashlib.sha256(idx).hexdigest()))


class Base(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="parley_test_install_")
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)
        self.root = self.tmp / "Parley"
        self.srv = FakeServer()
        self.addCleanup(self.srv.close)
        self.state = self.tmp / "update_state.json"

    def updater(self, current="0.1.0") -> Updater:
        return Updater(f"{self.srv.base}/parley/version.json", current, state_path=self.state,
                       _allow_http_localhost=True)

    def release(self, version, files, drop_unchanged_from=None):
        make_release(self.srv, version, files)
        if drop_unchanged_from:  # prove unchanged files are never downloaded
            for p, d in files.items():
                if drop_unchanged_from.get(p) == d:
                    del self.srv.routes[f"/parley/blobs/{sha(d)}"]
        u = self.updater()
        m = u.check()
        return u, m, u.fetch_index(m)

    def install(self, u, m, idx, **kw):
        kw.setdefault("self_check", lambda d: True)
        kw.setdefault("harden", lambda d: None)
        return install.install_version(self.root, u, m, idx, **kw)

    def pointer(self):
        return launcher.read_pointer(self.root / launcher.POINTER)


class InstallTests(Base):
    def test_success_flips_pointer_links_files_and_notes_verified(self):
        old = tree()
        seed(self.root, "0.1.0", old)
        new = tree(b"import sys\nsys.exit(0)\n# v2\n")
        u, m, idx = self.release("0.2.0", new, drop_unchanged_from=old)
        before = snapshot(launcher.app_path(self.root, "0.1.0"))
        self.install(u, m, idx)
        p = self.pointer()
        self.assertEqual((p.version, p.previous, p.state), ("0.2.0", "0.1.0", "pending"))
        d = launcher.app_path(self.root, "0.2.0")
        for rel, data in new.items():
            self.assertEqual((d / rel).read_bytes(), data)
        self.assertTrue(os.path.samefile(d / "data/same.bin", launcher.app_path(self.root, "0.1.0") / "data/same.bin"))
        self.assertEqual(snapshot(launcher.app_path(self.root, "0.1.0")), before)  # running version untouched
        self.assertFalse(list((self.root / "app").glob("*.partial")))
        self.assertEqual(json.loads(self.state.read_text())["highest_seen"], "0.2.0")
        launcher.tree_ok(self.root, p) or self.fail("pending tree must verify")

    def test_linked_files_are_read_only(self):
        seed(self.root, "0.1.0", tree())
        self.install(*self.release("0.2.0", tree(OK + b"#2")))
        with self.assertRaises(PermissionError):
            (launcher.app_path(self.root, "0.2.0") / "data/same.bin").write_bytes(b"oops")

    def test_confirm_and_prune_keeps_two_versions(self):
        seed(self.root, "0.1.0", tree())
        self.install(*self.release("0.2.0", tree(b"import sys\nsys.exit(0)\n#2")))
        self.assertTrue(install.confirm(self.root, "0.2.0"))
        self.assertEqual(self.pointer().state, "good")
        self.assertFalse(install.confirm(self.root, "0.2.0"))  # already good
        self.install(*self.release("0.3.0", tree(b"import sys\nsys.exit(0)\n#3")))
        self.assertFalse(install.confirm(self.root, "0.2.0"))  # not the pending version
        self.assertTrue(install.confirm(self.root, "0.3.0"))
        self.assertEqual(sorted(p.name for p in (self.root / "app").iterdir()), ["0.2.0", "0.3.0"])
        self.assertEqual(self.pointer().previous, "0.2.0")

    def assertUntouched(self, before, exc, call):
        with self.assertRaises(exc):
            call()
        self.assertEqual(snapshot(self.root), before)
        self.assertFalse(self.state.exists(), "note_verified must not run")

    def test_tampered_blob_aborts_before_flip(self):
        seed(self.root, "0.1.0", tree())
        u, m, idx = self.release("0.2.0", tree(b"import sys\nsys.exit(0)\n#2"))
        good = tree(b"import sys\nsys.exit(0)\n#2")["main.py"]
        self.srv.routes[f"/parley/blobs/{sha(good)}"] = b"X" * len(good)
        before = snapshot(self.root)
        self.assertUntouched(before, DownloadError, lambda: self.install(u, m, idx))

    def test_failure_mid_build_leaves_old_state_identical(self):
        seed(self.root, "0.1.0", tree())
        u, m, idx = self.release("0.2.0", tree(b"import sys\nsys.exit(0)\n#2", **{"zz/last.bin": b"last"}))
        del self.srv.routes[f"/parley/blobs/{sha(b'last')}"]  # 404 on the last file, after others were built
        before = snapshot(self.root)
        self.assertUntouched(before, httpx.HTTPStatusError, lambda: self.install(u, m, idx))

    def test_self_check_failure_aborts(self):
        seed(self.root, "0.1.0", tree())
        u, m, idx = self.release("0.2.0", tree(b"import sys\nsys.exit(0)\n#2"))
        before = snapshot(self.root)
        self.assertUntouched(before, InstallError, lambda: self.install(u, m, idx, self_check=lambda d: False))

    def test_tree_changed_during_self_check_aborts(self):
        seed(self.root, "0.1.0", tree())
        u, m, idx = self.release("0.2.0", tree(b"import sys\nsys.exit(0)\n#2"))
        before = snapshot(self.root)

        def evil(d: Path) -> bool:
            (d / "main.py").write_bytes(b"changed")
            return True
        self.assertUntouched(before, InstallError, lambda: self.install(u, m, idx, self_check=evil))

    def test_extra_file_in_tree_is_rejected(self):
        seed(self.root, "0.1.0", tree())
        u, m, idx = self.release("0.2.0", tree(b"import sys\nsys.exit(0)\n#2"))
        before = snapshot(self.root)

        def evil(d: Path) -> bool:
            (d / "extra.dll").write_bytes(b"x")
            return True
        self.assertUntouched(before, InstallError, lambda: self.install(u, m, idx, self_check=evil))

    def test_tampered_source_is_not_hardlinked(self):
        old = tree()
        seed(self.root, "0.1.0", old)
        u, m, idx = self.release("0.2.0", tree(b"import sys\nsys.exit(0)\n#2"))  # blobs for everything exist
        src = launcher.app_path(self.root, "0.1.0") / "data/same.bin"
        src.write_bytes(b"t" * 5000)  # same size, different content
        self.install(u, m, idx)
        dest = launcher.app_path(self.root, "0.2.0") / "data/same.bin"
        self.assertEqual(dest.read_bytes(), old["data/same.bin"])
        self.assertFalse(os.path.samefile(src, dest))
        self.assertEqual(src.read_bytes(), b"t" * 5000)

    def test_pointer_write_failure_leaves_old_state(self):
        seed(self.root, "0.1.0", tree())
        u, m, idx = self.release("0.2.0", tree(b"import sys\nsys.exit(0)\n#2"))
        before = snapshot(self.root)
        with mock.patch("parley.install.write_pointer", side_effect=OSError("disk full")):
            self.assertUntouched(before, OSError, lambda: self.install(u, m, idx))

    def test_hardlink_failure_falls_back_to_copy(self):
        seed(self.root, "0.1.0", tree())
        u, m, idx = self.release("0.2.0", tree(b"import sys\nsys.exit(0)\n#2"), drop_unchanged_from=tree())
        with mock.patch("parley.install.os.link", side_effect=OSError("no links")):
            self.install(u, m, idx)
        a = launcher.app_path(self.root, "0.1.0") / "data/same.bin"
        b = launcher.app_path(self.root, "0.2.0") / "data/same.bin"
        self.assertEqual(a.read_bytes(), b.read_bytes())
        self.assertFalse(os.path.samefile(a, b))

    def test_not_newer_or_skipped_refused(self):
        seed(self.root, "0.3.0", tree())
        u, m, idx = self.release("0.2.0", tree())
        before = snapshot(self.root)
        with self.assertRaises(InstallError):
            self.install(u, m, idx)
        seed(self.root, "0.1.0", tree())
        p = self.pointer()
        p.skip = ["0.2.0"]
        launcher.write_pointer(self.root, p)
        with self.assertRaises(InstallError):
            self.install(u, m, idx)
        self.assertTrue(install.is_skipped(self.root, "0.2.0"))
        self.assertFalse(list((self.root / "app").glob("*.partial")))
        self.assertTrue(before)

    def test_update_errors_tuple(self):
        for exc in (UpdateError, DownloadError, httpx.HTTPError, OSError, InstallError):
            self.assertTrue(issubclass(exc, install.UPDATE_ERRORS))

    @unittest.skipUnless(sys.platform == "win32", "icacls")
    def test_harden_dir_removes_group_access(self):
        d = self.tmp / "locked"
        install.harden_dir(d)
        out = subprocess.run(["icacls", str(d)], capture_output=True, text=True).stdout
        for broad in ("Everyone", "BUILTIN\\Users", "Authenticated Users"):
            self.assertNotIn(broad, out)
        (d / "x.txt").write_text("still usable")

    def test_child_processes_get_no_console_window(self):
        """A windowed parent must not flash a console for icacls or the self-check."""
        calls = []

        def fake_run(cmd, **kw):
            calls.append(kw)
            return subprocess.CompletedProcess(cmd, 0, "", "")

        flag = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        with mock.patch.object(install.subprocess, "run", fake_run),                 mock.patch.dict(os.environ, {"USERNAME": "tester"}):
            install.run_self_check(self.tmp)
            if sys.platform == "win32":
                install.harden_dir(self.tmp / "h")
        self.assertEqual(len(calls), 2 if sys.platform == "win32" else 1)
        for kw in calls:
            self.assertEqual(kw.get("creationflags"), flag)
        if sys.platform == "win32":
            self.assertNotEqual(flag, 0)

    def test_default_self_check_runs_exe(self):
        self.assertFalse(install.run_self_check(self.tmp))  # no exe there: False, not an exception


class DeadlineTests(Base):
    def test_overall_deadline_and_min_rate(self):
        u, m, idx = self.release("0.2.0", tree())
        with mock.patch.object(update, "MAX_DOWNLOAD_SECONDS", -1.0):
            with self.assertRaises(UpdateError):
                u.download_blob(idx[0], self.tmp / "a.part")
        with mock.patch.object(update, "RATE_GRACE", -1.0), mock.patch.object(update, "MIN_RATE", 10**12):
            with self.assertRaises(UpdateError):
                u.download_blob(idx[0], self.tmp / "b.part")
        self.assertEqual(u.download_blob(idx[0], self.tmp / "c.bin").read_bytes(), tree()[idx[0].path])


class TrickleTests(unittest.TestCase):
    def test_trickling_server_hits_deadline_quickly(self):
        import socket
        import threading
        import time
        ls = socket.socket()
        ls.bind(("127.0.0.1", 0))
        ls.listen(1)
        stop = threading.Event()

        def serve():
            c, _ = ls.accept()
            c.recv(4096)
            c.sendall(b"HTTP/1.1 200 OK" + CRLF + b"Content-Length: 100000" + CRLF + CRLF)
            try:
                while not stop.is_set():
                    c.sendall(b"x")
                    time.sleep(0.05)
            except OSError:
                pass
            c.close()
        threading.Thread(target=serve, daemon=True).start()
        self.addCleanup(ls.close)
        self.addCleanup(stop.set)
        u = Updater(f"http://127.0.0.1:{ls.getsockname()[1]}/parley/version.json", "0.1.0",
                    _allow_http_localhost=True)
        with tempfile.TemporaryDirectory(prefix="parley_test_install_") as d:
            f = update.IndexFile("a.bin", "0" * 64, 100000)
            t0 = time.monotonic()
            with mock.patch.object(update, "MAX_DOWNLOAD_SECONDS", 1.0):
                with self.assertRaises(UpdateError):
                    u.download_blob(f, Path(d) / "a.bin")
            self.assertLess(time.monotonic() - t0, 4.0)


class LauncherTests(Base):
    def go(self, **kw):
        return launcher.run(self.root, [], child=child, **kw)

    def install_pending(self, main: bytes):
        seed(self.root, "0.1.0", tree())
        self.install(*self.release("0.2.0", tree(main)))

    def test_good_version_runs_and_returns_code(self):
        seed(self.root, "0.1.0", tree(b"import sys\nsys.exit(7)\n"))
        self.assertEqual(self.go(), 7)
        self.assertEqual(self.pointer().state, "good")

    def test_crash_before_confirm_rolls_back_and_skips(self):
        self.install_pending(CRASH)
        self.assertEqual(self.go(), 0)  # the old version ran and exited 0
        p = self.pointer()
        self.assertEqual((p.version, p.state, p.skip, p.previous), ("0.1.0", "good", ["0.2.0"], None))
        self.assertTrue(install.is_skipped(self.root, "0.2.0"))

    def test_confirmed_version_is_kept_even_if_it_crashes_later(self):
        self.install_pending(OK)
        seen = []

        def runner(cmd):
            seen.append(cmd)
            install.confirm(self.root, "0.2.0")  # the app reached Ready
            return 1
        self.assertEqual(self.go(runner=runner), 1)
        self.assertEqual(self.pointer().version, "0.2.0")
        self.assertEqual(len(seen), 1)

    def test_unconfirmed_launches_roll_back_after_three(self):
        self.install_pending(OK)
        calls = []
        for _ in range(3):
            self.assertEqual(self.go(runner=lambda c: calls.append(c) or 0), 0)
        self.assertEqual((self.pointer().version, self.pointer().attempts), ("0.2.0", 3))
        self.go(runner=lambda c: calls.append(c) or 0)
        self.assertEqual(len(calls), 4)  # the fourth start was the old version
        self.assertIn("0.1.0", str(calls[-1]))
        self.assertEqual((self.pointer().version, self.pointer().skip), ("0.1.0", ["0.2.0"]))

    def test_pending_state_survives_a_kill(self):
        self.install_pending(OK)

        def killed(cmd):
            raise KeyboardInterrupt
        with self.assertRaises(KeyboardInterrupt):
            self.go(runner=killed)
        p = self.pointer()  # as a fresh process would see it
        self.assertEqual((p.version, p.state, p.attempts), ("0.2.0", "pending", 1))
        self.assertEqual(self.go(), 0)
        self.assertEqual(self.pointer().attempts, 2)

    def test_tampered_pending_tree_is_rolled_back_without_running(self):
        self.install_pending(OK)
        tamper(launcher.app_path(self.root, "0.2.0") / "data/other.txt", b"evil")
        ran = []
        self.go(runner=lambda c: ran.append(c) or 0)
        self.assertEqual(len(ran), 1)
        self.assertIn("0.1.0", str(ran[0]))
        self.assertEqual(self.pointer().skip, ["0.2.0"])

    def test_corrupt_pointer_falls_back_to_backup(self):
        self.install_pending(OK)
        (self.root / launcher.POINTER).write_text("{ not json")
        self.assertEqual(self.go(), 0)
        self.assertEqual(self.pointer().version, "0.1.0")  # the pre-update pointer
        (self.root / launcher.POINTER).write_bytes(b"\xff\x00")
        (self.root / launcher.POINTER_BACKUP).write_text("garbage")
        self.assertEqual(self.go(), 0)  # no backup either: newest app dir is used
        self.assertEqual(self.pointer().version, "0.2.0")

    def test_missing_current_dir_uses_previous(self):
        self.install_pending(OK)
        install._rmtree(launcher.app_path(self.root, "0.2.0"))
        ran = []
        self.go(runner=lambda c: ran.append(c) or 0)
        self.assertIn("0.1.0", str(ran[0]))
        self.assertEqual(self.pointer().skip, ["0.2.0"])

    def test_nothing_installed(self):
        with self.assertRaises(launcher.LauncherError):
            self.go()

    def test_pointer_parser_is_strict(self):
        for bad in ['[]', '{"version": "1.0"}', '{"version": "1.0.0", "state": "weird"}',
                    '{"version": "1.0.0", "attempts": true}', '{"version": "1.0.0", "skip": ["x"]}',
                    '{"version": "1.0.0", "index_sha256": "zz"}', '{"version": "1.0.0", "previous": 3}']:
            with self.assertRaises(ValueError, msg=bad):
                launcher.parse_pointer(bad)


if __name__ == "__main__":
    unittest.main()
