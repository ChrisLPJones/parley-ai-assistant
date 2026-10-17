"""scripts/release.py (stdlib unittest, temp dirs only): server layout, local publish order, refusals, ssh command
construction (no connection is ever made), notes, export of a ref. The real build is checked by hand (PROJECT.md)."""
import hashlib
import http.server
import importlib.util
import io
import json
import os
import shlex
import sys
import tempfile
import threading
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

from parley import launcher, update

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("parley_release_script", ROOT / "scripts" / "release.py")
rel = importlib.util.module_from_spec(_spec)
sys.modules["parley_release_script"] = rel
_spec.loader.exec_module(rel)

FILES = {"parley-app.exe": b"MZ" * 10, "_internal/python312.dll": b"d" * 3000, "_internal/pkg/data.json": b"{}"}


def fake_build(root: Path, version: str, files: dict[str, bytes] = FILES) -> tuple[Path, Path]:
    """A dist/install tree and installer like scripts/build.py + make_installer.py produce."""
    install = root / "install"
    app = launcher.app_path(install, version)
    entries = []
    for p, data in files.items():
        (app / p).parent.mkdir(parents=True, exist_ok=True)
        (app / p).write_bytes(data)
        entries.append({"path": p, "sha256": hashlib.sha256(data).hexdigest(), "size": len(data)})
    idx = json.dumps({"schema": 1, "version": version, "files": sorted(entries, key=lambda e: e["path"])}, sort_keys=True).encode()
    (app / launcher.INDEX_NAME).write_bytes(idx)
    launcher.write_pointer(install, launcher.Pointer(version, index_sha256=hashlib.sha256(idx).hexdigest()), backup=False)
    inst = root / f"Parley-Setup-{version}.exe"
    inst.write_bytes(b"installer " + version.encode())
    return install, inst


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="parley_test_release_unit_")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def layout(self, version="0.1.1", notes="Fixes things", files=FILES, sub="b"):
        install, inst = fake_build(self.root / sub, version, files)
        return rel.make_layout(install, inst, version, notes, released="2026-01-02")


class Notes(unittest.TestCase):
    def test_control_characters_stripped_newlines_kept(self):
        self.assertEqual(rel.clean_notes("  a\x00b\x1b[31m\r\nline2\tx\u202e \x07"), "ab[31m\nline2\tx")

    def test_too_long_refused(self):
        with self.assertRaises(rel.ReleaseError):
            rel.clean_notes("x" * (rel.MAX_NOTES + 1))

    def test_notes_file_and_conflict(self):
        with tempfile.TemporaryDirectory(prefix="parley_test_release_notes_") as d:
            p = Path(d, "RELEASE_NOTES")
            p.write_bytes(b"Line one\r\nLine two\x00")
            self.assertEqual(rel.read_notes(None, str(p)), "Line one\nLine two")
            with self.assertRaises(rel.ReleaseError):
                rel.read_notes("x", str(p))
            with self.assertRaises(rel.ReleaseError):
                rel.read_notes(None, str(Path(d, "missing")))
            big = Path(d, "big")
            big.write_bytes(b"x" * (rel.MAX_NOTES_FILE + 1))
            with self.assertRaises(rel.ReleaseError):
                rel.read_notes(None, str(big))
        self.assertEqual(rel.read_notes(None, None), "")


class Refusals(Base):
    def args(self, *extra):
        return rel.parse_args(["--notes", "n", *extra])

    def test_defaults_ok(self):
        self.assertEqual(rel.validate_args(self.args()), "n")
        self.assertEqual(rel.parse_args([]).ref, "origin/main")

    def test_allow_rejected_paths_refused(self):
        with self.assertRaisesRegex(rel.ReleaseError, "rejected-paths"):
            rel.validate_args(self.args("--allow-rejected-paths"))

    def test_http_feed_refused_for_publish(self):
        for url in ("http://updates.example.com/parley/version.json", "ftp://x/parley/version.json",
                    "https://u:p@x.com/parley/version.json", "https://x.com/parley/other.json", "https://x.com"):
            with self.assertRaises(rel.ReleaseError, msg=url):
                rel.validate_args(self.args("--feed-url", url))

    def test_http_localhost_only_for_local_test_target(self):
        local = "http://127.0.0.1:8123/parley/version.json"
        with self.assertRaises(rel.ReleaseError):
            rel.validate_args(self.args("--feed-url", local))  # no flag
        with self.assertRaises(rel.ReleaseError):
            rel.validate_args(self.args("--feed-url", local, "--test-http-feed"))  # ssh target
        with self.assertRaises(rel.ReleaseError):
            rel.validate_args(self.args("--feed-url", "http://example.com/parley/version.json", "--test-http-feed",
                                        "--target", str(self.root)))  # not localhost
        rel.validate_args(self.args("--feed-url", local, "--test-http-feed", "--target", str(self.root)))

    def test_version_override_only_with_local_target(self):
        with self.assertRaises(rel.ReleaseError):
            rel.validate_args(self.args("--version-override", "9.9.9"))
        with self.assertRaises(rel.ReleaseError):
            rel.validate_args(self.args("--version-override", "1.2", "--target", str(self.root)))
        rel.validate_args(self.args("--version-override", "9.9.9", "--target", str(self.root)))

    def test_bad_ssh_host_or_dir(self):
        for host in ("-oProxyCommand=x", "a b", "a;b", ""):
            with self.assertRaises(rel.ReleaseError, msg=host):
                rel.validate_args(self.args(f"--ssh-host={host}"))
        for d in ("relative", "/srv/../etc", "/srv/a b", "/srv/a;rm", "/srv/x/"):
            with self.assertRaises(rel.ReleaseError, msg=d):
                rel.validate_args(self.args(f"--remote-dir={d}"))

    def test_downgrade_and_equal_refused_unless_forced(self):
        rel.check_newer("0.1.2", "0.1.1", False)
        rel.check_newer("0.1.0", None, False)
        for new in ("0.1.1", "0.1.0", "0.0.9"):
            with self.assertRaises(rel.ReleaseError):
                rel.check_newer(new, "0.1.1", False)
            rel.check_newer(new, "0.1.1", True)

    def test_published_version_parse(self):
        self.assertIsNone(rel.parse_published_version(None))
        self.assertEqual(rel.parse_published_version(b'{"version": "1.2.3"}'), "1.2.3")
        for bad in (b"nope", b'{"version": "x"}', b"[]", b"{}", b"\xff\xfe"):
            with self.assertRaises(rel.ReleaseError):
                rel.parse_published_version(bad)
            with redirect_stderr(io.StringIO()):  # --force-republish tolerates a corrupt published manifest
                self.assertIsNone(rel.parse_published_version(bad, True))

    def test_local_corrupt_published_version_needs_force(self):
        Path(self.root, "version.json").write_bytes(b"{corrupt")
        with self.assertRaises(rel.ReleaseError):
            rel.read_local_version(self.root)
        with redirect_stderr(io.StringIO()):
            self.assertIsNone(rel.read_local_version(self.root, True))

    def test_unknown_and_option_like_refs(self):
        for ref in ("no-such-ref-xyz", "--upload-pack=x", "a b", ""):
            with self.assertRaises(rel.ReleaseError, msg=ref):
                rel.resolve_ref(ref)
        self.assertRegex(rel.resolve_ref("HEAD"), "^[0-9a-f]{40}$")

    def test_smoke_never_on_real_home(self):
        real = Path(os.environ.get("LOCALAPPDATA", str(self.root))) / "parley"
        self.assertFalse(rel.smoke_home_ok(real))
        self.assertFalse(rel.smoke_home_ok(self.root / "somewhere"))
        self.assertTrue(rel.smoke_home_ok(self.root / "parley_test_release"))
        with self.assertRaises(rel.ReleaseError):
            rel.smoke(self.root, "0.1.0", real)

    def test_tag_is_printed_never_created(self):
        self.assertEqual(rel.tag_command("1.2.3", "a" * 40), ["git", "tag", "v1.2.3", "a" * 40])
        calls = []
        real_run, real_popen = rel.subprocess.run, rel.subprocess.Popen

        def spy_run(argv, *a, **k):
            calls.append(list(argv))
            return real_run(argv, *a, **k)

        def spy_popen(argv, *a, **k):
            calls.append(list(argv))
            return real_popen(argv, *a, **k)
        with mock.patch.object(rel.subprocess, "run", spy_run), mock.patch.object(rel.subprocess, "Popen", spy_popen):
            sha = rel.resolve_ref("HEAD")
            rel.check_tag("0.0.1", sha)
            try:
                rel.require_on_main(sha)
            except rel.ReleaseError:
                pass  # HEAD may be a feature branch; only the git calls made matter
            rel.export_ref(sha, self.root / "e")
        self.assertTrue(calls)
        for argv in calls:
            self.assertEqual(argv[0], "git")
            sub = [x for x in argv[1:] if x not in ("-C", str(ROOT))][0]
            self.assertNotIn(sub, {"tag", "push", "commit", "checkout", "add", "reset", "merge", "branch", "stash"}, argv)

    def test_real_publish_fetches_origin_main_with_list_args_first(self):
        calls = []

        class P:
            returncode, stdout, stderr = 0, "", ""
        with mock.patch.object(rel.subprocess, "run", lambda argv, *a, **k: calls.append(list(argv)) or P()):
            rel.fetch_origin_main()
        self.assertEqual(calls[0][-3:], ["fetch", "origin", "main"])

        class Bad(P):
            returncode, stderr = 1, "no network"
        with mock.patch.object(rel.subprocess, "run", lambda *a, **k: Bad()), self.assertRaisesRegex(rel.ReleaseError, "stale"):
            rel.fetch_origin_main()
        order = []
        with mock.patch.object(rel, "fetch_origin_main", lambda: order.append("fetch")), \
                mock.patch.object(rel, "resolve_ref", lambda r: order.append("resolve") or (_ for _ in ()).throw(rel.ReleaseError("stop"))):
            with self.assertRaises(rel.ReleaseError):
                rel.run(["--notes", "n"])  # real ssh publish: fetch first
            self.assertEqual(order, ["fetch", "resolve"])
            order.clear()
            with self.assertRaises(rel.ReleaseError):
                rel.run(["--notes", "n", "--dry-run"])
            self.assertEqual(order, ["resolve"])  # dry run never touches the network

    def test_clean_env_drops_cuda(self):
        env = rel.clean_env({"PATH": os.pathsep.join(["C:\\x", "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.1\\bin",
                                                       "D:\\cuda\\bin", "C:\\y"]),
                             "CUDA_PATH": "c", "CUDA_PATH_V12_5": "c", "VIRTUAL_ENV": "v", "HOME": "h"})
        self.assertEqual(env["PATH"], os.pathsep.join(["C:\\x", "C:\\y"]))
        self.assertEqual(set(env) - {"PATH"}, {"HOME"})


class Export(Base):
    def test_export_of_head_is_tracked_files_only_and_override_stays_in_export(self):
        sha = rel.resolve_ref("HEAD")
        out = self.root / "export"
        rel.export_ref(sha, out)
        self.assertTrue((out / "scripts" / "build.py").is_file())
        self.assertFalse((out / ".git").exists())
        self.assertFalse((out / "src" / "parley" / "_build.py").exists())
        before = (ROOT / "src" / "parley" / "__init__.py").read_bytes()
        rel.override_version(out, "7.8.9")
        self.assertEqual(rel.read_version(out), "7.8.9")
        self.assertEqual((ROOT / "src" / "parley" / "__init__.py").read_bytes(), before)
        with self.assertRaises(rel.ReleaseError):  # not empty
            rel.export_ref(sha, out)

    def test_read_version_rejects_bad(self):
        d = self.root / "e" / "src" / "parley"
        d.mkdir(parents=True)
        (d / "__init__.py").write_text('__version__ = "1.x"\n', "utf-8")
        with self.assertRaises(rel.ReleaseError):
            rel.read_version(self.root / "e")

    def test_build_commands_are_list_args_with_feed(self):
        cmds = rel.build_commands("https://h/parley/version.json")
        self.assertTrue(all(isinstance(c, list) for c in cmds))
        self.assertIn("--frozen", cmds[0])
        self.assertEqual(cmds[1][-2:], ["--feed-url", "https://h/parley/version.json"])
        self.assertNotIn("--allow-rejected-paths", sum(cmds, []))


class LayoutAndPublish(Base):
    def test_layout_is_what_the_client_parses(self):
        lo = self.layout()
        m = update.parse_manifest(lo.version_json)
        self.assertEqual((m.version, m.notes, m.released), ("0.1.1", "Fixes things", "2026-01-02"))
        self.assertEqual(m.platform.index, "0.1.1/windows-x64/files.json")
        self.assertEqual(m.platform.sha256, hashlib.sha256(lo.installer.read_bytes()).hexdigest())
        self.assertEqual(m.platform.size, lo.installer.stat().st_size)
        self.assertIsNone(m.signature)
        doc = json.loads(lo.version_json)
        self.assertEqual((doc["file"], doc["sha256"]), (m.platform.file, m.platform.sha256))
        files = update.parse_index(lo.files_json, m.platform.index_sha256, "0.1.1")
        self.assertEqual({f.sha256 for f in files}, set(lo.blobs))
        for sha, src in lo.blobs.items():
            self.assertEqual(hashlib.sha256(src.read_bytes()).hexdigest(), sha)

    def test_files_json_equals_trees_index_bytes(self):
        lo = self.layout()
        self.assertEqual(lo.files_json, (self.root / "b" / "install" / "app" / "0.1.1" / launcher.INDEX_NAME).read_bytes())

    def test_refuses_tree_the_client_would_reject(self):
        install, inst = fake_build(self.root / "bad", "0.1.1", {**FILES, "voices/!v/x.bin": b"x"})
        with self.assertRaisesRegex(rel.ReleaseError, "not publishable"):
            rel.make_layout(install, inst, "0.1.1", "")

    def test_refuses_modified_tree_wrong_version_or_installer(self):
        install, inst = fake_build(self.root / "m", "0.1.1")
        (launcher.app_path(install, "0.1.1") / "parley-app.exe").write_bytes(b"tampered")
        with self.assertRaises(rel.ReleaseError):
            rel.make_layout(install, inst, "0.1.1", "")
        install, inst = fake_build(self.root / "w", "0.1.1")
        with self.assertRaises(rel.ReleaseError):
            rel.make_layout(install, inst, "0.1.2", "")
        inst.rename(inst.with_name("Other.exe"))
        with self.assertRaises(rel.ReleaseError):
            rel.make_layout(install, inst.with_name("Other.exe"), "0.1.1", "")

    def test_publish_local_writes_layout_with_version_json_last(self):
        lo = self.layout()
        target, steps = self.root / "pub", []
        rel.publish_local(lo, target, steps.append)
        self.assertEqual(steps[-1], "version.json")
        self.assertEqual(sum(s.startswith("blobs/") for s in steps), len(lo.blobs))
        self.assertLess(max(i for i, s in enumerate(steps) if s.startswith("blobs/")), steps.index("Parley-Setup-0.1.1.exe"))
        self.assertLess(steps.index("0.1.1/windows-x64/files.json"), steps.index("version.json"))
        for sha in lo.blobs:
            self.assertEqual(hashlib.sha256((target / "blobs" / sha).read_bytes()).hexdigest(), sha)
        self.assertEqual((target / "0.1.1" / "windows-x64" / "files.json").read_bytes(), lo.files_json)
        self.assertEqual((target / "version.json").read_bytes(), lo.version_json)
        self.assertFalse((target / "version.json.prev").exists())
        self.assertEqual([p.name for p in target.rglob(".*.tmp")], [])

    def test_version_json_only_appears_when_everything_else_is_in_place(self):
        lo = self.layout()
        target = self.root / "pub"
        seen = []

        def step(name):
            if name == "version.json":
                seen.append(all((target / "blobs" / s).is_file() for s in lo.blobs)
                            and (target / lo.installer_name).is_file() and (target / lo.files_json_path).is_file())
        rel.publish_local(lo, target, step)
        self.assertEqual(seen, [True])

    def test_failure_before_the_end_leaves_old_version_json(self):
        old = self.layout("0.1.0", sub="o")
        new = self.layout("0.1.1", sub="n")
        target = self.root / "pub"
        rel.publish_local(old, target)
        (new.installer).unlink()  # the copy of the installer fails after the blobs went in
        with self.assertRaises(OSError):
            rel.publish_local(new, target)
        self.assertEqual((target / "version.json").read_bytes(), old.version_json)

    def test_second_publish_keeps_prev_and_shares_blobs(self):
        a = self.layout("0.1.0", sub="a")
        b = self.layout("0.1.1", sub="b2", files={**FILES, "parley-app.exe": b"MZ-new"})
        target, steps = self.root / "pub", []
        rel.publish_local(a, target)
        self.assertEqual(rel.read_local_version(target), "0.1.0")
        rel.publish_local(b, target, steps.append)
        self.assertEqual(sum(s.startswith("blobs/") for s in steps), 1)  # only the changed file
        self.assertEqual((target / "version.json.prev").read_bytes(), a.version_json)
        self.assertEqual(rel.read_local_version(target), "0.1.1")
        with self.assertRaises(rel.ReleaseError):
            rel.check_newer("0.1.1", rel.read_local_version(target), False)


class Ssh(Base):
    def test_commands_are_list_args_in_order(self):
        cmds = rel.ssh_commands("0.1.1", "updates-deploy", "/srv/updates/parley", self.root)
        self.assertEqual(len(cmds), 5)
        for c in cmds:
            self.assertIsInstance(c.argv, list)
            self.assertTrue(all(isinstance(x, str) for x in c.argv))
        self.assertEqual([c.argv[0] for c in cmds], ["ssh", "scp", "ssh", "scp", "ssh"])
        for c in cmds:
            self.assertIn("BatchMode=yes", c.argv)
            self.assertIn("ConnectTimeout=10", c.argv)
            self.assertIn("StrictHostKeyChecking=yes", c.argv)
        self.assertEqual(cmds[1].argv[-2:], ["incoming-0.1.1", "updates-deploy:/srv/updates/parley/"])
        self.assertEqual(cmds[3].argv[-2:], ["version.json.tmp", "updates-deploy:/srv/updates/parley/version.json.tmp"])
        self.assertEqual(cmds[1].cwd, self.root)
        # the manifest is only moved live by the very last command
        self.assertIn("mv -f version.json.tmp version.json", cmds[4].argv[-1])
        self.assertIn("version.json.prev", cmds[4].argv[-1])
        for early in cmds[:4]:
            self.assertNotIn("mv -f version.json.tmp version.json", early.argv[-1])

    def test_remote_command_is_one_quoted_string(self):
        s = rel.remote(rel.REMOTE_MOVE, "/srv/a b/x", "0.1.1")
        parts = shlex.split(s)
        self.assertEqual(parts[:2], ["sh", "-c"])
        self.assertEqual(parts[2], rel.REMOTE_MOVE)
        self.assertEqual(parts[-2:], ["/srv/a b/x", "0.1.1"])  # data travels as positional args, never spliced into the script
        self.assertNotIn("/srv/a b/x", rel.REMOTE_MOVE)

    def test_every_remote_script_sets_umask_first(self):
        for script in (rel.REMOTE_READ_VERSION, rel.REMOTE_PREPARE, rel.REMOTE_MOVE, rel.REMOTE_COMMIT):
            self.assertTrue(script.startswith("umask 022; "), script)
        self.assertIn("StrictHostKeyChecking=yes", rel.SSH_OPTS)

    def test_unsafe_values_rejected(self):
        for host, root, ver in (("a;b", "/srv/x", "0.1.1"), ("h", "/srv/$(id)", "0.1.1"), ("h", "/srv/x", "0.1.1;rm -rf")):
            with self.assertRaises((rel.ReleaseError, ValueError), msg=(host, root, ver)):
                rel.ssh_commands(ver, host, root, self.root)

    def test_dry_run_prints_every_command_and_runs_none(self):
        lo = self.layout()

        def boom(*a, **k):
            raise AssertionError("dry run must not run anything")
        buf = io.StringIO()
        with redirect_stdout(buf):
            rel.publish_ssh(lo, "updates-deploy", "/srv/updates/parley", self.root / "w", dry_run=True, runner=boom)
        out = buf.getvalue()
        self.assertIn("DRY RUN", out)
        for c in rel.ssh_commands("0.1.1", "updates-deploy", "/srv/updates/parley", self.root / "w" / "upload"):
            self.assertIn(c.shown(), out)
        self.assertTrue((self.root / "w" / "upload" / "version.json.tmp").is_file())

    def test_real_run_order_skips_existing_blobs_and_is_stoppable(self):
        lo = self.layout()
        have = sorted(lo.blobs)[0]
        calls = []

        class R:
            def __init__(self, rc=0, out=""):
                self.returncode, self.stdout = rc, out

        def fake(argv, cwd=None, capture_output=False, text=False):
            calls.append(argv)
            if len(calls) == 1:
                return R(0, have + "\nnotablob\n")
            if len(calls) == 2:
                inc = Path(cwd, "incoming-0.1.1", "blobs")
                self.assertEqual({p.name for p in inc.iterdir()}, set(lo.blobs) - {have})
                # the manifest is staged for a later step but is not live: nothing has moved it into place yet
                self.assertTrue(Path(cwd, "version.json.tmp").is_file())
                self.assertFalse(Path(cwd, "version.json").exists())
            return R()
        buf = io.StringIO()
        with redirect_stdout(buf):
            rel.publish_ssh(lo, "updates-deploy", "/srv/updates/parley", self.root / "w", dry_run=False, runner=fake)
        self.assertEqual([c[0] for c in calls], ["ssh", "scp", "ssh", "scp", "ssh"])
        # failure of the upload stops before anything later (in particular the manifest) runs
        calls.clear()

        def failing(argv, cwd=None, capture_output=False, text=False):
            calls.append(argv)
            return R(1 if len(calls) == 2 else 0, "")
        with redirect_stdout(io.StringIO()), self.assertRaises(rel.ReleaseError):
            rel.publish_ssh(lo, "updates-deploy", "/srv/updates/parley", self.root / "w2", dry_run=False, runner=failing)
        self.assertEqual(len(calls), 2)


class Verify(Base):
    """Post-publish check against a local http.server (test-only http hook); never a real server."""

    def serve(self, body: bytes, status: int = 200):
        class H(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(status)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *a):
                pass
        srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        self.addCleanup(srv.server_close)
        self.addCleanup(srv.shutdown)
        return f"http://127.0.0.1:{srv.server_address[1]}/parley/version.json"

    def test_matching_bytes_verify(self):
        lo = self.layout()
        url = self.serve(lo.version_json)
        with redirect_stdout(io.StringIO()) as out:
            rel.verify_published(url, lo.version_json, _allow_http_localhost=True)
        self.assertIn("verified", out.getvalue())

    def test_different_bytes_or_error_fail_loudly(self):
        lo = self.layout()
        with self.assertRaisesRegex(rel.ReleaseError, "VERIFY FAILED"):
            rel.verify_published(self.serve(lo.version_json + b" "), lo.version_json, _allow_http_localhost=True)
        with self.assertRaisesRegex(rel.ReleaseError, "VERIFY FAILED"):
            rel.verify_published(self.serve(b"x", 404), lo.version_json, _allow_http_localhost=True)

    def test_http_refused_without_the_test_hook_and_args_checked(self):
        lo = self.layout()
        with self.assertRaisesRegex(rel.ReleaseError, "VERIFY FAILED"):
            rel.verify_published("http://127.0.0.1:9/parley/version.json", lo.version_json)
        with self.assertRaises(rel.ReleaseError):
            rel.validate_args(rel.parse_args(["--notes", "n", "--verify-url", "http://example.com/parley/version.json"]))

    def test_manual_instruction_has_expected_hash_and_no_curl(self):
        lo = self.layout()
        text = rel.verify_instruction("https://h/parley/version.json", lo.version_json)
        self.assertIn(hashlib.sha256(lo.version_json).hexdigest(), text)
        self.assertNotIn("curl", text)


if __name__ == "__main__":
    unittest.main()
