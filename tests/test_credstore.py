"""credstore: key precedence, migration out of config.toml, no key in logs / repr, --set-key, one real round trip.

Run: uv run python -m unittest discover -s tests -v
Scratch dirs are temp dirs named parley_test_cred_*; PARLEY_HOME is patched to one of them, so the real Parley home
and the real "Parley/*" credentials are never read or written. The one real Credential Manager test uses the
target prefix parley_test_<uuid> and deletes it in a finally block.
"""
import contextlib
import hashlib
import http.server
import io
import logging
import os
import sys
import tempfile
import threading
import unittest
import uuid
from pathlib import Path
from unittest import mock

from parley import config, credstore
from parley.credstore import CredentialError, CredStore

KEY = "sk-ant-" "api03-TESTKEY0123456789abcdef"
OAI = "sk-proj-TESTKEY0123456789abcdef"


class FakeBackend:
    def __init__(self):
        self.items: dict[str, str] = {}
        self.fail_write = self.fail_read = self.corrupt_write = False

    def read(self, target):
        if self.fail_read:
            raise CredentialError("read failed")
        return self.items.get(target)

    def write(self, target, blob):
        if self.fail_write:
            raise CredentialError("write failed")
        self.items[target] = "corrupted" if self.corrupt_write else blob

    def delete(self, target):
        return self.items.pop(target, None) is not None


class Capture(logging.Handler):
    def __init__(self):
        super().__init__(logging.DEBUG)
        self.lines: list[str] = []

    def emit(self, record):
        self.lines.append(self.format(record))

    @property
    def text(self):
        return "\n".join(self.lines)


class Base(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="parley_test_cred_")
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)
        env = mock.patch.dict(os.environ, {"PARLEY_HOME": str(self.dir)})
        env.start()
        self.addCleanup(env.stop)
        for var in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_BASE_URL"):
            os.environ.pop(var, None)
        self.backend = FakeBackend()
        self.store = CredStore(self.backend)
        self.path = self.dir / "config.toml"
        self.logs = Capture()
        root = logging.getLogger()
        self._level = root.level
        root.setLevel(logging.DEBUG)
        root.addHandler(self.logs)
        self.addCleanup(lambda: (root.removeHandler(self.logs), root.setLevel(self._level)))
        credstore._warned.clear()
        credstore._secrets.clear()


class TargetTests(Base):
    def test_target_carries_home_hash(self):
        tag = hashlib.sha256(os.path.normcase(os.path.abspath(str(self.dir))).encode()).hexdigest()[:8]
        self.assertEqual(self.store.target("anthropic"), f"Parley-{tag}/anthropic")
        self.assertRegex(self.store.target("openai"), r"^Parley-[0-9a-f]{8}/openai$")

    def test_no_home_means_plain_prefix(self):
        with mock.patch.dict(os.environ):
            os.environ.pop("PARLEY_HOME")
            self.assertEqual(self.store.target("openai"), "Parley/openai")

    def test_two_homes_do_not_share_keys(self):
        self.store.set("anthropic", KEY)
        with mock.patch.dict(os.environ, {"PARLEY_HOME": str(self.dir / "other")}):
            self.assertIsNone(self.store.get("anthropic"))

    def test_unknown_backend_and_bad_keys_refused(self):
        with self.assertRaises(CredentialError):
            self.store.target("builtin")
        for bad in ("", "has space", "café-key", "x" * 513, "nul\0key", "new\nline"):
            with self.assertRaises(CredentialError) as cm:
                self.store.set("openai", bad)
            self.assertNotIn(bad.strip() or "zzz", str(cm.exception).replace("API key", ""))

    def test_unavailable_store(self):
        store = CredStore(credstore.UnavailableBackend())
        self.assertFalse(store.available)
        self.assertIsNone(store.get("anthropic"))
        self.assertFalse(store.exists("anthropic"))
        self.assertFalse(store.delete("anthropic"))
        with self.assertRaises(CredentialError):
            store.set("anthropic", KEY)

    def test_functions_use_default_store(self):
        with mock.patch.object(credstore, "_default", self.store):
            credstore.set("openai", OAI)
            self.assertEqual(credstore.get("openai"), OAI)
            self.assertTrue(credstore.exists("openai"))
            self.assertTrue(credstore.delete("openai"))
            self.assertFalse(credstore.exists("openai"))


class PrecedenceTests(Base):
    def test_table(self):
        for stored in (False, True):
            for env in (False, True):
                for legacy in (False, True):
                    with self.subTest(stored=stored, env=env, legacy=legacy), mock.patch.dict(os.environ):
                        self.backend.items.clear()
                        os.environ.pop("ANTHROPIC_API_KEY", None)
                        if stored:
                            self.store.set("anthropic", "stored-key-123")
                        if env:
                            os.environ["ANTHROPIC_API_KEY"] = "env-key-123"
                        cfg = config.LlmConfig(api_key="legacy-key-123" if legacy else "")
                        want = "stored-key-123" if stored else "env-key-123" if env else "legacy-key-123" if legacy else ""
                        self.assertEqual(credstore.resolve_api_key("anthropic", cfg, self.store), want)

    def test_only_source_is_logged(self):
        self.store.set("openai", "stored-key-123")
        credstore.resolve_api_key("openai", config.LlmConfig(), self.store)
        self.assertIn("credential store", self.logs.text)
        self.assertNotIn("stored-key-123", self.logs.text)

    def test_unreadable_store_falls_through_with_warning(self):
        self.backend.fail_read = True
        os.environ["OPENAI_API_KEY"] = "env-key-123"
        self.assertEqual(credstore.resolve_api_key("openai", config.LlmConfig(), self.store), "env-key-123")
        self.assertIn("Could not read", self.logs.text)

    def test_env_var_is_per_backend(self):
        os.environ["OPENAI_API_KEY"] = "env-key-123"
        self.assertEqual(credstore.resolve_api_key("anthropic", config.LlmConfig(), self.store), "")


CONFIG = """# my notes
[hotkey]
key = "pause"  # hold to talk

[llm]
backend = "anthropic"
api_key = "%s"  # old
model = "x"
"""


class MigrationTests(Base):
    def write(self, text=None, backend="anthropic", key=KEY):
        self.path.write_text(text if text is not None else CONFIG % key, encoding="utf-8", newline="")

    def test_success_removes_line_keeps_comments(self):
        self.write()
        self.assertEqual(credstore.migrate_legacy_key(self.path, self.store), credstore.MIGRATED)
        text = self.path.read_text(encoding="utf-8")
        self.assertNotIn(KEY, text)
        self.assertNotIn("api_key", text)
        for keep in ("# my notes", "# hold to talk", 'model = "x"', 'backend = "anthropic"'):
            self.assertIn(keep, text)
        self.assertEqual(self.store.get("anthropic"), KEY)
        self.assertNotIn(KEY, self.logs.text)
        self.assertEqual(config.load(self.path).llm.api_key, "")

    def test_idempotent(self):
        self.write()
        credstore.migrate_legacy_key(self.path, self.store)
        after = self.path.read_bytes()
        self.assertEqual(credstore.migrate_legacy_key(self.path, self.store), credstore.NOTHING)
        self.assertEqual(self.path.read_bytes(), after)
        self.assertEqual(self.store.get("anthropic"), KEY)

    def test_failures_leave_config_byte_identical(self):
        for flag in ("fail_write", "fail_read", "corrupt_write"):
            with self.subTest(flag):
                self.backend.items.clear()
                self.write()
                before = self.path.read_bytes()
                setattr(self.backend, flag, True)
                self.assertEqual(credstore.migrate_legacy_key(self.path, self.store), credstore.FAILED)
                setattr(self.backend, flag, False)
                self.assertEqual(self.path.read_bytes(), before)
                self.assertNotIn(KEY, self.logs.text)
                self.assertEqual(self.backend.items, {}, "half-written entry must be undone")
                self.assertFalse(self.path.with_name("config.toml.bak").exists())

    def test_config_write_failure_undoes_store_entry(self):
        self.write()
        before = self.path.read_bytes()
        with mock.patch("parley.config_io.os.replace", side_effect=OSError("disk full")):
            self.assertEqual(credstore.migrate_legacy_key(self.path, self.store), credstore.FAILED)
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual(self.backend.items, {})

    def test_ambiguous_key_left_in_place(self):
        for backend in ("builtin", "ollama", "server"):
            with self.subTest(backend):
                self.write(f'[llm]\nbackend = "{backend}"\napi_key = "weird-key-123456"\n')
                before = self.path.read_bytes()
                self.assertEqual(credstore.migrate_legacy_key(self.path, self.store), credstore.LEFT_IN_PLACE)
                self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual(self.backend.items, {})

    def test_server_backend_key_never_moved_even_with_sk_prefix(self):
        self.write(f'[llm]\nbackend = "server"\napi_key = "{OAI}"\n')
        self.assertEqual(credstore.migrate_legacy_key(self.path, self.store), credstore.LEFT_IN_PLACE)

    def test_target_by_prefix_for_other_backends(self):
        self.write(f'[llm]\nbackend = "builtin"\napi_key = "{KEY}"\n')
        self.assertEqual(credstore.migrate_legacy_key(self.path, self.store), credstore.MIGRATED)
        self.assertEqual(self.store.get("anthropic"), KEY)
        self.write(f'[llm]\nbackend = "ollama"\napi_key = "{OAI}"\n')
        self.assertEqual(credstore.migrate_legacy_key(self.path, self.store), credstore.MIGRATED)
        self.assertEqual(self.store.get("openai"), OAI)

    def test_configured_backend_wins_over_prefix(self):
        self.write(f'[llm]\nbackend = "openai"\napi_key = "{KEY}"\n')
        credstore.migrate_legacy_key(self.path, self.store)
        self.assertEqual(self.store.get("openai"), KEY)
        self.assertIsNone(self.store.get("anthropic"))

    def test_different_stored_key_is_not_overwritten(self):
        self.store.set("anthropic", "newer-stored-key-1")
        self.write()
        before = self.path.read_bytes()
        self.assertEqual(credstore.migrate_legacy_key(self.path, self.store), credstore.LEFT_IN_PLACE)
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual(self.store.get("anthropic"), "newer-stored-key-1")

    def test_same_stored_key_just_removes_the_line(self):
        self.store.set("anthropic", KEY)
        self.write()
        self.assertEqual(credstore.migrate_legacy_key(self.path, self.store), credstore.MIGRATED)
        self.assertNotIn(KEY, self.path.read_text(encoding="utf-8"))
        self.assertEqual(self.store.get("anthropic"), KEY)

    def test_read_only_config_is_untouched(self):
        self.write()
        before = self.path.read_bytes()
        os.chmod(self.path, 0o444)
        self.addCleanup(lambda: os.chmod(self.path, 0o644))
        self.assertEqual(credstore.migrate_legacy_key(self.path, self.store), credstore.FAILED)
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual(self.backend.items, {})
        credstore.migrate_legacy_key(self.path, self.store)
        self.assertEqual(self.logs.text.count("read-only or locked"), 1, "warn once")

    def test_unparsable_config_is_untouched_and_warns_once(self):
        self.write("[llm\napi_key = oops\n")
        before = self.path.read_bytes()
        for _ in range(2):
            self.assertEqual(credstore.migrate_legacy_key(self.path, self.store), credstore.FAILED)
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual(self.logs.text.count("unreadable or has a mistake"), 1)

    def test_no_file_or_no_key(self):
        self.assertEqual(credstore.migrate_legacy_key(self.path, self.store), credstore.NOTHING)
        self.write('[llm]\nbackend = "anthropic"\napi_key = ""\n')
        self.assertEqual(credstore.migrate_legacy_key(self.path, self.store), credstore.NOTHING)

    def test_unavailable_store_leaves_key(self):
        self.write()
        before = self.path.read_bytes()
        self.assertEqual(credstore.migrate_legacy_key(self.path, CredStore(credstore.UnavailableBackend())), credstore.LEFT_IN_PLACE)
        self.assertEqual(self.path.read_bytes(), before)

    def test_bak_holding_key_is_deleted_after_success(self):
        self.write()
        bak = self.path.with_name("config.toml.bak")
        bak.write_text(CONFIG % KEY, encoding="utf-8")
        self.assertEqual(credstore.migrate_legacy_key(self.path, self.store), credstore.MIGRATED)
        self.assertFalse(bak.exists())

    def test_bak_kept_when_migration_fails(self):
        self.write()
        bak = self.path.with_name("config.toml.bak")
        bak.write_text(CONFIG % KEY, encoding="utf-8")
        self.backend.fail_write = True
        credstore.migrate_legacy_key(self.path, self.store)
        self.assertTrue(bak.exists())

    def test_unrelated_bak_is_kept_and_no_bak_is_created(self):
        self.write()
        bak = self.path.with_name("config.toml.bak")
        bak.write_text('[hotkey]\nkey = "f13"\n', encoding="utf-8")
        credstore.migrate_legacy_key(self.path, self.store)
        self.assertTrue(bak.exists())
        self.assertNotIn(KEY, bak.read_text(encoding="utf-8"))
        bak.unlink()
        self.write()
        credstore.migrate_legacy_key(self.path, CredStore(FakeBackend()))
        self.assertFalse(bak.exists(), "migration must not create a .bak while the key is in the file")


class NoLeakTests(Base):
    def test_repr_of_config_and_exceptions(self):
        cfg = config.Config()
        cfg.llm.api_key = KEY
        self.assertNotIn(KEY, repr(cfg))
        self.assertNotIn(KEY, str(cfg))
        try:
            credstore.CredStore(self.backend).set("openai", "bad key")
        except CredentialError as e:
            self.assertNotIn("bad key", str(e))

    def test_providers_repr(self):
        from parley.claude import ClaudeProvider
        from parley.llm import OpenAICompatibleProvider

        with mock.patch.object(credstore, "_default", self.store):
            self.store.set("anthropic", KEY)
            p = ClaudeProvider(config.LlmConfig(), None, False)
        self.assertNotIn(KEY, repr(p) + repr(p.client) + repr(vars(p)))
        q = OpenAICompatibleProvider(config.LlmConfig(), None, "http://127.0.0.1:9/v1", OAI, "m", cloud=True)
        self.assertNotIn(OAI, repr(q) + repr(q.client) + repr(vars(q).get("cfg")))

    def test_debug_logging_of_a_real_httpx_request_has_no_key(self):
        from parley.llm import OpenAICompatibleProvider

        class H(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                body = b'{"data": []}'
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *a):
                pass

        srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        self.addCleanup(srv.server_close)
        self.addCleanup(srv.shutdown)
        names = ("httpx", "httpcore", "anthropic", "openai")
        old = {n: logging.getLogger(n).level for n in names}
        for n in names:
            logging.getLogger(n).setLevel(logging.DEBUG)
        try:
            q = OpenAICompatibleProvider(config.LlmConfig(), None, f"http://127.0.0.1:{srv.server_port}/v1", OAI, "m", cloud=True)
            q.client.get("/models")
            q.client.close()
        finally:
            for n, lvl in old.items():
                logging.getLogger(n).setLevel(lvl)
        self.assertTrue(self.logs.lines, "expected debug lines from httpx")
        self.assertNotIn(OAI, self.logs.text)

    def test_filter_scrubs_message_and_traceback(self):
        credstore.register_secret("plain-secret-value")
        h = Capture()
        h.addFilter(credstore.RedactFilter())
        h.setFormatter(logging.Formatter("%(message)s"))
        lg = logging.getLogger("parley.test_leak")
        lg.addHandler(h)
        self.addCleanup(lg.removeHandler, h)
        lg.warning("headers: %s", {"Authorization": f"Bearer {OAI}", "x": "plain-secret-value"})
        try:
            raise RuntimeError(f"boom {KEY}")
        except RuntimeError:
            lg.exception("failed")
        self.assertNotIn(OAI, h.text)
        self.assertNotIn(KEY, h.text)
        self.assertNotIn("plain-secret-value", h.text)
        self.assertIn("RuntimeError", h.text)

    def test_custom_endpoint_warning_names_only_host(self):
        credstore.warn_if_custom_endpoint("openai", "https://user:pw@proxy.example.com:8443/v1?token=abc")
        self.assertIn("proxy.example.com", self.logs.text)
        for leak in ("user", "pw", "8443", "token", "abc"):
            self.assertNotIn(leak, self.logs.text.replace("proxy.example.com", ""), leak)
        self.logs.lines.clear()
        credstore.warn_if_custom_endpoint("openai", "https://api.openai.com/v1")
        credstore.warn_if_custom_endpoint("server", "http://127.0.0.1:8080/v1", loopback_ok=True)
        self.assertEqual(self.logs.lines, [])


class CliTests(Base):
    def run_set(self, key, backend="anthropic", store=None):
        out, err = io.StringIO(), io.StringIO()
        with mock.patch("getpass.getpass", return_value=key) as gp:
            code = credstore.cli_set_key(backend, store or self.store, out, err)
        return code, out.getvalue(), err.getvalue(), gp

    def test_set_key_stores_without_echo(self):
        code, out, err, gp = self.run_set(KEY)
        self.assertEqual(code, 0)
        self.assertEqual(self.store.get("anthropic"), KEY)
        self.assertNotIn(KEY, out + err)
        self.assertNotIn(KEY, gp.call_args.args[0])

    def test_set_key_rejects_empty_and_bad_without_saving(self):
        for bad in ("", "   ", "has space inside", "café"):
            code, out, err, _ = self.run_set(bad)
            self.assertEqual(code, 1, bad)
            self.assertNotIn("inside", out + err)
        self.assertEqual(self.backend.items, {})

    def test_set_key_store_failure_and_unavailable(self):
        self.backend.fail_write = True
        code, out, err, _ = self.run_set(KEY)
        self.assertEqual(code, 1)
        self.assertNotIn(KEY, out + err)
        code, out, err, gp = self.run_set(KEY, store=CredStore(credstore.UnavailableBackend()))
        self.assertEqual(code, 1)
        gp.assert_not_called()

    def test_delete_and_status(self):
        self.run_set(KEY)
        cfg = config.Config()
        cfg.llm.backend, cfg.llm.api_key = "openai", "legacy-key-123"
        os.environ["OPENAI_API_KEY"] = "env-key-123"
        out = io.StringIO()
        credstore.cli_key_status(cfg, self.store, out)
        text = out.getvalue()
        self.assertIn("anthropic: Saved; in use: credential store", text)
        self.assertIn("openai: Not saved; in use: environment variable", text)
        for leak in (KEY, "legacy-key-123", "env-key-123"):
            self.assertNotIn(leak, text)
        out, err = io.StringIO(), io.StringIO()
        self.assertEqual(credstore.cli_delete_key("anthropic", self.store, out, err), 0)
        self.assertFalse(self.store.exists("anthropic"))
        self.assertIn("Deleted", out.getvalue())
        out = io.StringIO()
        credstore.cli_delete_key("anthropic", self.store, out, err)
        self.assertIn("No saved", out.getvalue())


class RealCredentialManagerTests(unittest.TestCase):
    def test_round_trip(self):
        if sys.platform != "win32":
            self.skipTest("Windows only")
        try:
            backend = credstore.WinBackend()
        except ImportError:
            self.skipTest("win32cred not available")
        store = CredStore(backend, prefix=f"parley_test_{uuid.uuid4().hex}")
        try:
            try:
                store.set("anthropic", KEY)
            except CredentialError:
                self.skipTest("no usable credential store on this machine")
            self.assertEqual(store.get("anthropic"), KEY)
            self.assertTrue(store.exists("anthropic"))
            store.set("anthropic", OAI)  # replace
            self.assertEqual(store.get("anthropic"), OAI)
            self.assertTrue(store.delete("anthropic"))
            self.assertIsNone(store.get("anthropic"))
            self.assertFalse(store.delete("anthropic"))
        finally:
            with contextlib.suppress(Exception):
                store.delete("anthropic")


if __name__ == "__main__":
    unittest.main()
