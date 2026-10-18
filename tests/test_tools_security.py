"""Tool hardening: UNC paths, SSRF in read_webpage, the `open` allowlist, secret files, tool logging.
No real network, SMB or ShellExecute: DNS and the HTTP transport are stubbed; PARLEY_HOME is a throwaway dir."""
import logging
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import httpx

from parley import tools

HOME = Path(tempfile.gettempdir()) / "parley_test_toolsec"


def dns(mapping):
    return lambda host: mapping[host]


def client_for(handler, seen=None):
    def make():
        def wrapped(request):
            if seen is not None:
                seen.append(request)
            return handler(request)
        return httpx.Client(transport=httpx.MockTransport(wrapped), follow_redirects=False)
    return make


class Base(unittest.TestCase):
    def setUp(self):
        HOME.mkdir(exist_ok=True)
        p = mock.patch.dict(os.environ, {"PARLEY_HOME": str(HOME)})
        p.start()
        self.addCleanup(p.stop)
        self.tmp = tempfile.TemporaryDirectory(prefix="parley_test_toolsec_files_")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)


class UncPaths(Base):
    VARIANTS = [r"\\host\share\x", "//host/share/x", r"\\?\UNC\host\share\x", r"\\.\pipe\x", r"\/host/x", r"/\host/x",
                r'"\\host\share"']

    def test_refused_before_touching_filesystem(self):
        for v in self.VARIANTS:
            for fn in (tools.read_file, tools.list_folder, lambda p: tools.find_files("a", p), tools.open_item):
                with self.subTest(v=v), mock.patch("os.startfile", create=True) as sf, \
                        mock.patch("pathlib.Path.resolve", side_effect=AssertionError("touched fs")):
                    with self.assertRaises(PermissionError):
                        fn(v)
                    sf.assert_not_called()

    def test_resolving_to_unc_refused(self):
        with mock.patch("pathlib.Path.resolve", return_value=Path(r"\\evil\share\f.txt")):
            with self.assertRaises(PermissionError):
                tools.resolve(str(self.root / "link"))

    def test_env_expansion_to_unc_refused(self):
        with mock.patch.dict(os.environ, {"EVIL": r"\\host\share"}):
            with self.assertRaises(PermissionError):
                tools.resolve("%EVIL%" if os.name == "nt" else "$EVIL")

    def test_normal_path_ok(self):
        f = self.root / "a.txt"
        f.write_text("hi")
        self.assertEqual(tools.resolve(str(f)), f.resolve())


class WebpageSsrf(Base):
    def read(self, url, mapping, handler, seen=None):
        with mock.patch.object(tools, "_resolve_host", dns(mapping)), \
                mock.patch.object(tools, "_http_client", client_for(handler, seen)):
            return tools.read_webpage(url)

    def ok(self, request):
        return httpx.Response(200, html="<html><body><p>Hello page</p></body></html>")

    def test_public_page_pinned_to_validated_ip(self):
        seen = []
        out = self.read("https://example.com/x", {"example.com": ["93.184.216.34"]}, self.ok, seen)
        self.assertIn("Hello page", out)
        self.assertEqual(seen[0].url.host, "93.184.216.34")
        self.assertEqual(seen[0].headers["host"], "example.com")
        self.assertEqual(seen[0].extensions["sni_hostname"], "example.com")

    def test_blocked_addresses(self):
        for ip in ("127.0.0.1", "::1", "10.1.2.3", "192.168.0.5", "172.16.0.1", "169.254.169.254", "0.0.0.0", "::",
                   "::ffff:127.0.0.1", "fe80::1", "fc00::1", "224.0.0.1", "240.0.0.1", "100.64.0.1"):
            with self.subTest(ip=ip), self.assertRaises(PermissionError):
                self.read("http://evil.example/", {"evil.example": [ip]}, self.ok)

    def test_literal_ip_urls_blocked(self):
        for url in ("http://127.0.0.1:8080/", "http://[::1]/", "http://169.254.169.254/latest"):
            with self.subTest(url=url), mock.patch.object(tools, "_resolve_host", lambda h: [h]), \
                    self.assertRaises(PermissionError):
                tools.read_webpage(url)

    def test_any_bad_address_blocks(self):
        with self.assertRaises(PermissionError):
            self.read("http://mixed.example/", {"mixed.example": ["93.184.216.34", "10.0.0.1"]}, self.ok)

    def test_resolver_order_preserved(self):
        with mock.patch.object(tools.socket, "getaddrinfo", return_value=[
                (0, 0, 0, "", ("2606:2800::1", 0)), (0, 0, 0, "", ("93.184.216.34", 0)),
                (0, 0, 0, "", ("2606:2800::1", 0))]):
            self.assertEqual(tools._resolve_host("x.example"), ["2606:2800::1", "93.184.216.34"])

    def test_falls_back_to_next_address_on_connect_failure(self):
        seen = []

        def handler(request):
            if request.url.host == "2606:2800::1":
                raise httpx.ConnectError("unreachable", request=request)
            return httpx.Response(200, html="<p>Via v4</p>")

        out = self.read("http://dual.example/", {"dual.example": ["2606:2800::1", "93.184.216.34"]}, handler, seen)
        self.assertIn("Via v4", out)
        self.assertEqual([r.url.host for r in seen], ["2606:2800::1", "93.184.216.34"])

    def test_http_error_does_not_fall_back(self):
        seen = []
        with self.assertRaises(httpx.HTTPStatusError):
            self.read("http://dual.example/", {"dual.example": ["93.184.216.34", "93.184.216.35"]},
                      lambda r: httpx.Response(500), seen)
        self.assertEqual(len(seen), 1)

    def test_all_addresses_failing_raises_connect_error(self):
        def handler(request):
            raise httpx.ConnectError("down", request=request)

        with self.assertRaises(httpx.ConnectError):
            self.read("http://dual.example/", {"dual.example": ["2606:2800::1", "93.184.216.34"]}, handler)

    def test_private_address_after_public_one_refused_without_connecting(self):
        seen = []
        with self.assertRaises(PermissionError):
            self.read("http://mixed.example/", {"mixed.example": ["2606:2800::1", "10.0.0.1"]}, self.ok, seen)
        self.assertEqual(seen, [])

    def test_idn_host_uses_ascii_form(self):
        seen = []
        out = self.read("http://bücher.de/x", {"xn--bcher-kva.de": ["93.184.216.34"]}, self.ok, seen)
        self.assertIn("Hello page", out)
        self.assertEqual(seen[0].headers["host"], "xn--bcher-kva.de")
        self.assertEqual(seen[0].extensions["sni_hostname"], "xn--bcher-kva.de")

    def test_bad_schemes(self):
        for url in ("file:///C:/x", "ftp://example.com/", "gopher://example.com/"):
            with self.subTest(url=url), self.assertRaises(PermissionError):
                self.read(url, {"example.com": ["93.184.216.34"]}, self.ok)

    def test_redirect_to_private_blocked(self):
        seen = []

        def handler(request):
            if request.headers["host"] == "good.example":
                return httpx.Response(302, headers={"location": "http://internal.example/admin"})
            return httpx.Response(200, text="secret")

        with self.assertRaises(PermissionError):
            self.read("http://good.example/", {"good.example": ["93.184.216.34"], "internal.example": ["10.0.0.9"]},
                      handler, seen)
        self.assertEqual(len(seen), 1)  # the private hop was never requested

    def test_redirect_to_loopback_literal_blocked(self):
        def handler(request):
            return httpx.Response(301, headers={"location": "http://127.0.0.1:8080/v1/models"})

        with mock.patch.object(tools, "_resolve_host", lambda h: {"good.example": ["93.184.216.34"]}.get(h, [h])), \
                mock.patch.object(tools, "_http_client", client_for(handler)), self.assertRaises(PermissionError):
            tools.read_webpage("http://good.example/")

    def test_relative_redirect_followed_and_final_url_reported(self):
        def handler(request):
            if request.url.path == "/a":
                return httpx.Response(302, headers={"location": "/b"})
            return httpx.Response(200, html="<p>Landed</p>")

        out = self.read("http://good.example/a", {"good.example": ["93.184.216.34"]}, handler)
        self.assertIn("Landed", out)
        self.assertIn("http://good.example/b", out)

    def test_too_many_redirects(self):
        def handler(request):
            return httpx.Response(302, headers={"location": "/loop"})

        with self.assertRaises(RuntimeError):
            self.read("http://good.example/", {"good.example": ["93.184.216.34"]}, handler)

    def test_size_cap(self):
        chunks = []

        def gen():
            for _ in range(100):
                chunks.append(1)
                yield b"<p>" + b"x" * 100_000 + b"</p>"

        def handler(request):
            return httpx.Response(200, headers={"content-type": "text/html"}, stream=_Gen(gen()))

        self.read("http://good.example/", {"good.example": ["93.184.216.34"]}, handler)
        self.assertLessEqual(len(chunks), 25)  # 2 MB / ~100 KB, not all 100 (10 MB)

    def test_http_error_raises(self):
        with self.assertRaises(httpx.HTTPStatusError):
            self.read("http://good.example/", {"good.example": ["93.184.216.34"]}, lambda r: httpx.Response(404))


class _Gen(httpx.SyncByteStream):
    def __init__(self, it):
        self.it = it

    def __iter__(self):
        return iter(self.it)


class OpenAllowlist(Base):
    def open_file(self, name):
        f = self.root / name
        f.write_text("x")
        with mock.patch("os.startfile", create=True) as sf, mock.patch("subprocess.Popen") as popen:
            out = tools.open_item(str(f))
            return out, sf.called or popen.called

    def test_allowed_types_open(self):
        for name in ("a.txt", "a.PDF", "a.docx", "a.png", "a.mp3", "a.mkv"):
            with self.subTest(name=name):
                out, started = self.open_file(name)
                self.assertTrue(started)
                self.assertTrue(out.startswith("Opened"))

    def test_other_types_refused(self):
        for name in ("a.pif", "a.msc", "a.chm", "a.application", "a.appinstaller", "a.msix", "a.appx", "a.ws", "a.wsc",
                     "a.scf", "a.search-ms", "a.library-ms", "a.searchConnector-ms", "a.iso", "a.img", "a.vhd",
                     "a.vhdx", "a.diagcab", "a.xll", "a.exe", "a.lnk", "a.ps1", "a.url", "a.zip", "noext", "a.weird"):
            with self.subTest(name=name):
                out, started = self.open_file(name)
                self.assertFalse(started)
                self.assertIn("won't open it", out)

    def test_folder_still_opens(self):
        with mock.patch("os.startfile", create=True) as sf, mock.patch("subprocess.Popen") as popen:
            out = tools.open_item(str(self.root))
        self.assertTrue(out.startswith("Opened"))
        self.assertTrue(sf.called or popen.called)

    def test_website_unchanged(self):
        with mock.patch("webbrowser.open") as wb:
            out = tools.open_item("zzz-no-such-site.com")
        wb.assert_called_once_with("https://zzz-no-such-site.com")
        self.assertIn("Opened", out)


class SecretFiles(Base):
    def test_extended_blocklist(self):
        for name in (".netrc", "_netrc", ".npmrc", ".pypirc", "ConsoleHost_history.txt", "logins.json",
                     "sitemanager.xml"):
            with self.subTest(name=name), self.assertRaises(PermissionError):
                tools._check_allowed(self.root / name)

    def test_parley_config_blocked_only_in_parley_home(self):
        for name in ("config.toml", "config.toml.bak"):
            with self.subTest(name=name), self.assertRaises(PermissionError):
                tools._check_allowed(HOME / name)
        tools._check_allowed(self.root / "config.toml")  # someone else's config.toml is fine

    def test_read_file_blocks_parley_config(self):
        (HOME / "config.toml").write_text("[llm]\n")
        with self.assertRaises(PermissionError):
            tools.read_file(str(HOME / "config.toml"))

    def test_find_files_skips_secrets(self):
        (self.root / ".ssh").mkdir()
        (self.root / ".ssh" / "notes.txt").write_text("k")
        (self.root / "sub").mkdir()
        (self.root / "sub" / "notes.txt").write_text("ok")
        (self.root / "sub" / ".npmrc").write_text("token")
        (self.root / "sub" / "my.pem").write_text("k")
        import json
        out = json.loads(tools.find_files("*", str(self.root)))["matches"]
        # Display paths are relative to the user's home, so don't depend on the base.
        names = [m.replace("\\", "/") for m in out]
        self.assertTrue(any(n.endswith("sub/notes.txt") for n in names), names)
        self.assertFalse([n for n in names if ".ssh" in n or n.endswith((".npmrc", ".pem"))], names)

    def test_find_files_in_secret_root_refused(self):
        (self.root / ".ssh").mkdir()
        with self.assertRaises(PermissionError):
            tools.find_files("*", str(self.root / ".ssh"))

    def test_recursive_list_skips_secret_folders(self):
        import json
        (self.root / ".ssh").mkdir()
        (self.root / ".ssh" / "secretfile").write_text("k")
        (self.root / "ok.txt").write_text("1")
        out = json.loads(tools.list_folder(str(self.root), True))
        self.assertEqual(out["files"], 1)
        self.assertNotIn(".ssh", out["folder_names"])
        self.assertNotIn("secretfile", out["file_names"])


class Logging(Base):
    def test_arguments_and_results_not_logged(self):
        box = tools.Toolbox([tools.Tool("t", "d", {"q": tools.STR}, ["q"], lambda q: "RESULT-TEXT-SECRET")])
        with self.assertLogs("parley.tools", logging.INFO) as cm:
            box.call("t", {"q": "ARG-SECRET"})
        text = "\n".join(cm.output)
        self.assertNotIn("ARG-SECRET", text)
        self.assertNotIn("RESULT-TEXT-SECRET", text)
        self.assertIn("Tool t -> ok", text)


if __name__ == "__main__":
    unittest.main()
