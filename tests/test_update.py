"""Update client core against a local http.server fixture (stdlib unittest; no network, no Parley home).

Run: uv run python -m unittest discover -s tests -v
"""
import hashlib
import json
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from parley import update
from parley.download import DownloadError
from parley.update import UpdateError, Updater, parse_manifest
from parley.version import compare_versions, parse_version


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


class FakeServer:
    """Serves `routes` {path: bytes | (status, headers, body)}; `truncate` paths announce more than they send."""

    def __init__(self) -> None:
        self.routes: dict[str, object] = {}
        self.truncate: set[str] = set()
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a): pass

            def do_GET(self):
                route = outer.routes.get(self.path)
                if route is None:
                    self.send_error(404)
                    return
                status, headers, body = route if isinstance(route, tuple) else (200, {}, route)
                self.send_response(status)
                for k, v in headers.items():
                    self.send_header(k, v)
                n = len(body) + (10 if self.path in outer.truncate else 0)
                self.send_header("Content-Length", str(n))
                self.end_headers()
                self.wfile.write(body)
                self.close_connection = True

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()


def make_release(srv: FakeServer, version="0.2.0", files=None, **over):
    """Publish installer, index, blobs and manifest on the fixture; returns the manifest dict."""
    installer = b"INSTALLER-" + version.encode()
    files = files if files is not None else {"app/a.bin": b"aaa", "b.txt": b"bbb"}
    index = json.dumps({"schema": 1, "version": version, "files": [
        {"path": p, "sha256": sha(d), "size": len(d)} for p, d in files.items()]}).encode()
    srv.routes[f"/parley/{version}/windows-x64/files.json"] = index
    srv.routes[f"/parley/Parley-Setup-{version}.exe"] = installer
    for d in files.values():
        srv.routes[f"/parley/blobs/{sha(d)}"] = d
    entry = {"file": f"Parley-Setup-{version}.exe", "sha256": sha(installer), "size": len(installer),
             "index": f"{version}/windows-x64/files.json", "index_sha256": sha(index)}
    manifest = {"schema": 1, "version": version, "notes": "Fixes.", "released": "2026-10-17",
                "platforms": {"windows-x64": entry}, "signature": None, **over}
    srv.routes["/parley/version.json"] = json.dumps(manifest).encode()
    return manifest


class VersionTests(unittest.TestCase):
    def test_compare_table(self):
        for a, b, want in [("0.1.0", "0.1.0", 0), ("0.1.1", "0.1.0", 1), ("0.2.0", "0.10.0", -1),
                           ("1.0.0", "0.99.99", 1), ("0.0.9", "0.0.10", -1)]:
            self.assertEqual(compare_versions(a, b), want, (a, b))

    def test_rejects_non_semver(self):
        for bad in ["1.0", "1.0.0.0", "v1.0.0", "1.0.0-rc1", "01.0.0", "1.0.x", "", " 1.0.0", "1.0.0\n",
                    "\u0661.0.0", None, 1, "9" * 20 + ".0.0"]:
            with self.assertRaises(ValueError, msg=repr(bad)):
                parse_version(bad)


class ParseTests(unittest.TestCase):
    def good(self, **over):
        entry = {"file": "Setup.exe", "sha256": "a" * 64, "size": 10}
        doc = {"schema": 1, "version": "0.2.0", "platforms": {"windows-x64": entry}, **over}
        return json.dumps(doc).encode()

    def test_good_and_signature_parsed(self):
        m = parse_manifest(self.good(signature="abc", notes="Hi\x00\x1b[31m there"))
        self.assertEqual((m.version, m.signature, m.platform.file), ("0.2.0", "abc", "Setup.exe"))
        self.assertNotIn("\x00", m.notes)
        self.assertNotIn("\x1b", m.notes)

    def test_legacy_top_level_is_windows(self):
        doc = {"schema": 1, "version": "0.2.0", "file": "S.exe", "sha256": "b" * 64, "size": 5}
        self.assertEqual(parse_manifest(json.dumps(doc).encode()).platform.file, "S.exe")
        with self.assertRaises(UpdateError):
            parse_manifest(json.dumps(doc).encode(), "linux-x64")

    def test_missing_platform_rejected(self):
        with self.assertRaises(UpdateError):
            parse_manifest(self.good(), "linux-x64")

    def test_installer_name_rules(self):
        for name in ["NUL", "CON", "nul.exe", "setup", "setup.exe.txt", "a b.exe", ".exe", "con .exe"]:
            doc = {"schema": 1, "version": "0.2.0", "platforms": {"windows-x64": {
                "file": name, "sha256": "a" * 64, "size": 1}}}
            with self.assertRaises(UpdateError, msg=name):
                parse_manifest(json.dumps(doc).encode())

    def test_bad_input(self):
        cases = [b"not json", b"[]", b"\xff\xfe", b"{" * 5000, b"x" * (update.MAX_MANIFEST_BYTES + 1),
                 self.good(schema=2), self.good(schema=True), self.good(version="1.0"),
                 self.good(version="../1.0.0"), self.good(signature=5), self.good(platforms=[]),
                 self.good(platforms={"windows-x64": {"file": "../evil.exe", "sha256": "a" * 64, "size": 1}}),
                 self.good(platforms={"windows-x64": {"file": "a/b.exe", "sha256": "a" * 64, "size": 1}}),
                 self.good(platforms={"windows-x64": {"file": "C:evil.exe", "sha256": "a" * 64, "size": 1}}),
                 self.good(platforms={"windows-x64": {"file": "a.exe", "sha256": "A" * 64, "size": 1}}),
                 self.good(platforms={"windows-x64": {"file": "a.exe", "sha256": "a" * 64, "size": -1}}),
                 self.good(platforms={"windows-x64": {"file": "a.exe", "sha256": "a" * 64, "size": 1.5}}),
                 self.good(platforms={"windows-x64": {"file": "a.exe", "sha256": "a" * 64, "size": 1,
                                                      "index": "https://evil/x.json", "index_sha256": "a" * 64}}),
                 self.good(platforms={"windows-x64": {"file": "a.exe", "sha256": "a" * 64, "size": 1,
                                                      "index": "x/files.json"}})]
        for raw in cases:
            with self.assertRaises(UpdateError, msg=raw[:60]):
                parse_manifest(raw)


class PathTests(unittest.TestCase):
    def test_bad_paths(self):
        for p in ["../x", "a/../x", "/abs", "C:/x", "C:x", "a\\b", "a//b", "a/./b", "", ".", "x/", "con", "dir/NUL.txt",
                  "a/com1", "trail.", "sp ace /x", "a:stream", "q?", "x\x00y", 5, None, "a" * 300,
                  "con .txt", "nul .x", "COM¹", "lpt².txt", "x/aux.", "a\x7fb", "a‮b", "a b",
                  "café", "a b", "a!b", "a+ b", "a+:b", "a%b", "a​b"]:
            with self.assertRaises(UpdateError, msg=repr(p)):
                update.validate_rel_path(p)

    def test_good_paths(self):
        for p in ["a.exe", "app/lib/x.dll", "_internal/b-c_d.pyd", "con2/x",
                  "_internal/av.libs/libstdc++-6-abc.dll", "tzdata/zoneinfo/Etc/GMT+5", "pkg-1.0+local.dist-info/METADATA"]:
            self.assertEqual(update.validate_rel_path(p), p)

    def entry(self, path, data=b"x"):
        return {"path": path, "sha256": sha(data), "size": len(data)}

    def test_collisions(self):
        for files in ([self.entry("A.txt"), self.entry("a.TXT")],
                      [self.entry("a"), self.entry("a/b")],
                      [self.entry("a/b"), self.entry("A")],
                      [self.entry("x"), self.entry("x")]):
            with self.assertRaises(UpdateError, msg=files):
                update.validate_index_files(files)

    def test_limits_and_shapes(self):
        for bad in ([], "x", [1], [{"path": "a", "sha256": "z", "size": 1}],
                    [{"path": "a", "sha256": "a" * 64, "size": -1}],
                    [{"path": "a", "sha256": "a" * 64, "size": update.MAX_FILE_BYTES + 1}],
                    [{"path": f"f{i}", "sha256": "a" * 64, "size": update.MAX_FILE_BYTES} for i in range(3)]):
            with self.assertRaises(UpdateError, msg=bad):
                update.validate_index_files(bad)

    def test_check_reparse_free(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self.assertEqual(update.check_reparse_free(root, "a/b.txt"), root.resolve() / "a" / "b.txt")
            with self.assertRaises(UpdateError):
                update.check_reparse_free(root, "../b.txt")


class ClientTests(unittest.TestCase):
    def setUp(self):
        self.srv = FakeServer()
        self.addCleanup(self.srv.close)
        self.tmp = tempfile.TemporaryDirectory(prefix="parley_test_update_")
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)

    def updater(self, current="0.1.0", **kw):
        kw.setdefault("state_path", self.dir / "state.json")
        return Updater(f"{self.srv.base}/parley/version.json", current, _allow_http_localhost=True, **kw)

    def test_good_flow(self):
        make_release(self.srv)
        u = self.updater()
        m = u.check()
        self.assertEqual(m.version, "0.2.0")
        files = u.fetch_index(m)
        self.assertEqual({f.path for f in files}, {"app/a.bin", "b.txt"})
        for f in files:
            self.assertEqual(u.download_blob(f, self.dir / "blobs" / f.sha256).stat().st_size, f.size)
        inst = u.download_installer(m, self.dir)
        self.assertEqual(inst.read_bytes(), b"INSTALLER-0.2.0")
        self.assertFalse(list(self.dir.rglob("*.part")))

    def test_tampered_hash_and_size(self):
        make_release(self.srv)
        u = self.updater()
        m = u.check()
        f = u.fetch_index(m)[0]
        self.srv.routes[f"/parley/blobs/{f.sha256}"] = b"x" * f.size  # same size, other content
        with self.assertRaises(DownloadError):
            u.download_blob(f, self.dir / "blob")
        self.srv.routes[f"/parley/blobs/{f.sha256}"] = b"x" * (f.size + 5)
        with self.assertRaises(DownloadError):
            u.download_blob(f, self.dir / "blob")
        self.assertFalse((self.dir / "blob").exists())
        self.assertFalse((self.dir / "blob.part").exists())

    def test_tampered_index(self):
        make_release(self.srv)
        u = self.updater()
        m = u.check()
        self.srv.routes["/parley/0.2.0/windows-x64/files.json"] = b'{"schema":1,"version":"0.2.0","files":[]}'
        with self.assertRaises(UpdateError):
            u.fetch_index(m)

    def test_truncated_download(self):
        make_release(self.srv)
        u = self.updater()
        m = u.check()
        self.srv.truncate.add(f"/parley/{m.platform.file}")
        with self.assertRaises(Exception):  # httpx.RemoteProtocolError or DownloadError
            u.download_installer(m, self.dir)
        self.assertFalse((self.dir / m.platform.file).exists())
        self.assertFalse((self.dir / (m.platform.file + ".part")).exists())

    def test_downgrade_and_equal(self):
        make_release(self.srv, "0.2.0")
        self.assertIsNone(self.updater("0.2.0").check())
        self.assertIsNone(self.updater("0.3.0").check())
        self.assertIsNotNone(self.updater("0.1.9").check())

    def test_highest_seen_replay_protection(self):
        make_release(self.srv, "0.3.0")
        u = self.updater()
        self.assertEqual(u.check().version, "0.3.0")
        self.assertIsNone(self.updater().load_state().highest_seen)
        u.note_verified("0.3.0")
        self.assertEqual(u.load_state().highest_seen, "0.3.0")
        make_release(self.srv, "0.2.0")  # replayed older (still newer than installed)
        self.assertIsNone(u.check())
        make_release(self.srv, "0.3.0")  # equal
        self.assertIsNone(u.check())
        make_release(self.srv, "0.3.1")
        self.assertEqual(u.check().version, "0.3.1")
        u.note_verified("0.2.5")  # lower never lowers the floor
        self.assertEqual(u.load_state().highest_seen, "0.3.0")

    def test_corrupt_state_is_ignored(self):
        (self.dir / "state.json").write_text("{nope", "utf-8")
        make_release(self.srv)
        self.assertEqual(self.updater().check().version, "0.2.0")

    def test_bad_json_and_oversize_are_quiet(self):
        self.srv.routes["/parley/version.json"] = b"{{{"
        self.assertIsNone(self.updater().check())
        self.srv.routes["/parley/version.json"] = b" " * (update.MAX_MANIFEST_BYTES + 100)
        self.assertIsNone(self.updater().check())
        self.srv.routes["/parley/version.json"] = (500, {}, b"boom")
        self.assertIsNone(self.updater().check())

    def test_oversize_index(self):
        make_release(self.srv)
        u = self.updater()
        m = u.check()
        self.srv.routes["/parley/0.2.0/windows-x64/files.json"] = b" " * (update.MAX_INDEX_BYTES + 1)
        with self.assertRaises(UpdateError):
            u.fetch_index(m)

    def test_traversal_in_index_rejected(self):
        files = {"ok.txt": b"1"}
        make_release(self.srv, files=files)
        bad = json.dumps({"schema": 1, "version": "0.2.0", "files": [
            {"path": "../../evil.dll", "sha256": sha(b"1"), "size": 1}]}).encode()
        m = json.loads(self.srv.routes["/parley/version.json"])
        m["platforms"]["windows-x64"]["index_sha256"] = sha(bad)  # attacker controls hashes too
        self.srv.routes["/parley/version.json"] = json.dumps(m).encode()
        self.srv.routes["/parley/0.2.0/windows-x64/files.json"] = bad
        u = self.updater()
        with self.assertRaises(UpdateError):
            u.fetch_index(u.check())

    def test_offline_is_quiet(self):
        u = self.updater()
        self.srv.close()
        self.assertIsNone(u.check())
        self.assertTrue(u.last_check_failed)  # so the manual check can say "offline", not "up to date"

    def test_up_to_date_is_not_a_failed_check(self):
        make_release(self.srv, "0.1.0")
        u = self.updater()
        self.assertIsNone(u.check())  # the feed answered: the newest version is not newer than ours
        self.assertFalse(u.last_check_failed)

    def test_feed_url_needs_path(self):
        for url in ["https://example.com", "https://example.com/"]:
            with self.assertRaises(UpdateError, msg=url):
                Updater(url, "0.1.0")

    def test_download_verified_aborts_on_excess_bytes(self):
        from contextlib import contextmanager
        from parley.download import download_verified

        class Resp:
            def iter_bytes(self, n):
                yield b"a" * 6
                yield b"b" * 6
                raise AssertionError("kept reading after the size limit")

        @contextmanager
        def opener(url):
            yield Resp()

        dest = self.dir / "x.bin"
        with self.assertRaises(DownloadError):
            download_verified(opener, "u", dest, sha(b"a" * 6), 10)
        self.assertFalse(dest.exists())
        self.assertFalse((self.dir / "x.bin.part").exists())

    def test_https_only(self):
        with self.assertRaises(UpdateError):
            Updater("http://example.com/parley/version.json", "0.1.0")
        with self.assertRaises(UpdateError):  # test hook allows loopback only
            Updater("http://example.com/parley/version.json", "0.1.0", _allow_http_localhost=True)
        with self.assertRaises(UpdateError):
            Updater("http://127.0.0.1:1/version.json", "0.1.0")  # localhost http needs the hook
        with self.assertRaises(UpdateError):
            Updater("ftp://localhost/version.json", "0.1.0", _allow_http_localhost=True)
        with self.assertRaises(UpdateError):
            Updater("https://user:pw@example.com/version.json", "0.1.0")
        Updater("https://example.com/parley/version.json", "0.1.0")  # fine

    def test_same_origin_redirect_followed(self):
        make_release(self.srv)
        real = self.srv.routes["/parley/version.json"]
        self.srv.routes["/parley/version.json"] = (302, {"Location": "/parley/moved.json"}, b"")
        self.srv.routes["/parley/moved.json"] = real
        self.assertEqual(self.updater().check().version, "0.2.0")

    def test_cross_origin_and_downgrade_redirects_refused(self):
        other = FakeServer()
        self.addCleanup(other.close)
        make_release(other)
        for target in (f"{other.base}/parley/version.json", "https://127.0.0.1/parley/version.json",
                       "file:///etc/passwd", "//evil.example/x"):
            self.srv.routes["/parley/version.json"] = (302, {"Location": target}, b"")
            self.assertIsNone(self.updater().check(), target)

    def test_redirect_loop_capped(self):
        self.srv.routes["/parley/version.json"] = (302, {"Location": "/parley/version.json"}, b"")
        self.assertIsNone(self.updater().check())

    def test_build_info_absent_in_checkout(self):
        self.assertIsNone(update.build_info())
        self.assertIsNone(update.updater_from_build(self.dir / "s.json"))


if __name__ == "__main__":
    unittest.main()
