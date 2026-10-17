"""download(): progress, cancel and resume against a local http.server (no network, temp dirs parley_test_dl_*)."""
import http.server
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

from parley import download

PAYLOAD = bytes(range(256)) * (4 * 4096)  # 4 MiB


class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a): pass

    def do_GET(self):
        rng = self.headers.get("Range")
        start = int(rng.split("=")[1].split("-")[0]) if rng else 0
        body = PAYLOAD[start:]
        self.send_response(206 if rng else 200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        try:
            self.wfile.write(body)
        except OSError:
            pass


class DownloadProgressTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()
        cls.url = f"http://127.0.0.1:{cls.server.server_port}/f.bin"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="parley_test_dl_")
        self.dir = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        p = mock.patch.object(download, "PROGRESS_INTERVAL", 0)  # report on every chunk
        p.start()
        self.addCleanup(p.stop)

    def test_old_signature_still_works(self):
        dest = download.download(self.url, self.dir / "a.bin")
        self.assertEqual(dest.read_bytes(), PAYLOAD)

    def test_progress_reports_and_ends_complete(self):
        seen = []
        download.download(self.url, self.dir / "a.bin", progress=lambda d, t: seen.append((d, t)))
        self.assertGreater(len(seen), 1)
        self.assertEqual(seen[-1], (len(PAYLOAD), len(PAYLOAD)))
        self.assertEqual([d for d, _ in seen], sorted(d for d, _ in seen))
        self.assertTrue(all(t == len(PAYLOAD) for _, t in seen[:-1]))

    def test_cancel_keeps_part_and_resume_completes(self):
        cancel = threading.Event()
        dest = self.dir / "a.bin"
        with self.assertRaises(download.DownloadCancelled):
            download.download(self.url, dest, progress=lambda d, t: cancel.set(), cancel=cancel)
        part = dest.with_name("a.bin.part")
        self.assertTrue(part.exists())
        self.assertFalse(dest.exists())
        self.assertTrue(0 < part.stat().st_size < len(PAYLOAD))
        download.download(self.url, dest)  # resumes with a Range request
        self.assertEqual(dest.read_bytes(), PAYLOAD)
        self.assertFalse(part.exists())

    def test_observer_gets_file_name(self):
        seen = []
        download.set_observer(lambda n, d, t: seen.append((n, d, t)))
        self.addCleanup(download.set_observer, None)
        download.download(self.url, self.dir / "model.gguf")
        self.assertEqual(seen[-1], ("model.gguf", len(PAYLOAD), len(PAYLOAD)))

    def test_broken_progress_callback_does_not_fail_the_download(self):
        def bad(d, t):
            raise RuntimeError("x")
        dest = download.download(self.url, self.dir / "a.bin", progress=bad)
        self.assertEqual(dest.read_bytes(), PAYLOAD)

    def test_works_without_stderr(self):
        with mock.patch.object(download.sys, "stderr", None):
            self.assertEqual(download.download(self.url, self.dir / "a.bin").read_bytes(), PAYLOAD)


if __name__ == "__main__":
    unittest.main()
