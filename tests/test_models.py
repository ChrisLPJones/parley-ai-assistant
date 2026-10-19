"""Model catalog, fit verdicts and the download manager (security-sensitive: downloads and deletes).

Run: PARLEY_HOME=<temp>/parley_test_models uv run python -m unittest discover -s tests
No network: downloads come from a fake stream or a local http.server. Scratch dirs are temp dirs named parley_test_models_*.
"""
import hashlib
import http.server
import os
import tempfile
import threading
import unittest
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from unittest import mock

from parley import cuda_pack, download, llama_server, models
from parley.config import Config
from parley.download import DownloadCancelled, DownloadError, download_verified
from parley.models import CATALOG, File, Hardware, Model, ModelError, ModelManager

COMMIT = "a" * 40
DATA = bytes(range(256)) * 6000  # ~1.5 MiB, above the catalog's minimum model size
HOSTS = frozenset({"huggingface.co"})


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def hf_url(name: str, commit: str = COMMIT) -> str:
    return f"https://huggingface.co/x/y/resolve/{commit}/{name}"


def make_model(files: dict[str, bytes] | None = None, *, kind="llm", mid="llm-test", url=hf_url, hosts=HOSTS,
               subdir="", probe=("a.gguf",)) -> Model:
    files = {"a.gguf": DATA} if files is None else files
    fs = tuple(File(n, url(n), sha(b), len(b)) for n, b in files.items())
    return Model(mid, kind, "Test model", "", hosts, 2 * 1024**2, 1.0, "test", probe, fs, subdir)


class Resp:
    def __init__(self, data: bytes, status=200, headers=None, on_chunk=None):
        self.data, self.status_code, self.headers, self.on_chunk = data, status, headers or {}, on_chunk

    def iter_bytes(self, n=None):
        for i in range(0, len(self.data), 65536):
            if self.on_chunk:
                self.on_chunk(i)
            yield self.data[i:i + 65536]


class FakeNet:
    """open_stream(url, headers=None) over fixed bytes, with switches for a misbehaving server."""

    def __init__(self, data: bytes = DATA, *, mode="ok", on_chunk=None):
        self.data, self.mode, self.on_chunk, self.requests = data, mode, on_chunk, []

    def __call__(self, url, headers=None):
        self.requests.append((url, dict(headers or {})))
        rng = (headers or {}).get("Range")
        start = int(rng.split("=")[1].rstrip("-")) if rng else 0
        if rng and self.mode == "ignore-range":
            resp = Resp(self.data, 200, {}, self.on_chunk)
        elif rng and self.mode == "bad-content-range":
            resp = Resp(self.data[start:], 206, {"content-range": f"bytes 0-{len(self.data) - 1}/{len(self.data)}"}, self.on_chunk)
        elif rng:
            resp = Resp(self.data[start:], 206, {"content-range": f"bytes {start}-{len(self.data) - 1}/{len(self.data)}"},
                        self.on_chunk)
        else:
            resp = Resp(self.data, 200, {}, self.on_chunk)

        @contextmanager
        def ctx():
            yield resp
        return ctx()


def make_link(testcase: unittest.TestCase, link: Path, target: Path) -> None:
    """A directory symlink, or on Windows a junction (needs no privilege); skips the test if neither can be made."""
    try:
        os.symlink(target, link, target_is_directory=True)
        return
    except (OSError, NotImplementedError):
        pass
    try:
        import _winapi
        _winapi.CreateJunction(str(target), str(link))
    except (ImportError, OSError, AttributeError):
        testcase.skipTest("cannot create symlinks or junctions here")


class Scratch(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="parley_test_models_")
        self.addCleanup(self.tmp.cleanup)
        self.home = Path(self.tmp.name)

    def manager(self, model: Model, net=None, **kw) -> ModelManager:
        kw.setdefault("free_space", lambda p: 100 * 1024**3)
        return ModelManager(self.home, {model.id: model}, opener_factory=(lambda hosts: net) if net else None, **kw)

    def run_download(self, mgr: ModelManager, mid: str):
        mgr.download(mid)
        mgr.wait(30)
        return mgr.state


class CatalogTests(unittest.TestCase):
    def test_real_catalog_entries_are_valid_and_complete(self):
        for m in CATALOG.values():
            models.validate_model(m)
        self.assertEqual(len(CATALOG), len(models.SPECS))
        self.assertEqual(sorted(m.key for m in CATALOG.values() if m.kind == "llm"), ["medium", "small", "tiny"])
        self.assertEqual(sorted(m.key for m in CATALOG.values() if m.kind == "stt"),
                         ["base.en", "large-v3-turbo", "medium", "small", "small.en", "tiny.en"])
        self.assertEqual(sorted(m.key for m in CATALOG.values() if m.kind == "tts"), ["default", "fp16", "int8"])

    def test_every_preset_is_in_the_catalog_and_the_picked_one_too(self):
        for key, (_, file, _) in llama_server.PRESETS.items():
            self.assertEqual(CATALOG[f"llm-{key}"].probe, (file,))
        for plan_preset in ("tiny", "small"):
            self.assertIn(f"llm-{plan_preset}", CATALOG)
        with mock.patch.object(llama_server.gpu, "current_plan", return_value=mock.Mock(preset="small")):
            self.assertIn(f"llm-{llama_server.pick_preset()}", CATALOG)

    def test_pending_pin_entries_are_exactly_those_without_pins(self):
        self.assertEqual(set(models.pending()), {mid for mid in CATALOG if mid not in models.PINS})
        for mid in models.pending():
            self.assertFalse(CATALOG[mid].pinned)

    def test_pinned_entries_use_the_commit_never_main_and_unique_hashes(self):
        pins = {"llm-tiny": {"revision": COMMIT, "files": [("Qwen3.5-2B-Q4_K_M.gguf", sha(b"1"), 2_000_000)]},
                "stt-small.en": {"revision": COMMIT, "files": [("model.bin", sha(b"2"), 3_000_000),
                                                               ("config.json", sha(b"3"), 2_000)]},
                "tts-kokoro": {"revision": "model-files-v1.0", "files": [("kokoro-v1.0.onnx", sha(b"4"), 9_000_000)]}}
        cat = models.build_catalog(pins)
        for mid in pins:
            models.validate_model(cat[mid])
            self.assertTrue(cat[mid].pinned)
        self.assertEqual(cat["llm-tiny"].files[0].url,
                         f"https://huggingface.co/unsloth/Qwen3.5-2B-GGUF/resolve/{COMMIT}/Qwen3.5-2B-Q4_K_M.gguf")
        self.assertTrue(cat["tts-kokoro"].files[0].url.startswith("https://github.com/"))
        self.assertNotIn("/main/", " ".join(f.url for m in cat.values() for f in m.files))
        shas = [f.sha256 for m in cat.values() for f in m.files]
        self.assertEqual(len(shas), len(set(shas)))
        self.assertFalse(cat["llm-small"].pinned)

    def test_bad_entries_are_rejected(self):
        good = make_model()
        models.validate_model(good)
        f = good.files[0]
        bad = {
            "http": replace(good, files=(replace(f, url="http://huggingface.co/x"),)),
            "userinfo": replace(good, files=(replace(f, url="https://u:p@huggingface.co/x"),)),
            "foreign host": replace(good, files=(replace(f, url="https://evil.example/x"),)),
            "lookalike": replace(good, files=(replace(f, url="https://huggingface.co.evil.example/x"),)),
            "port": replace(good, files=(replace(f, url="https://huggingface.co:8443/x"),)),
            "dotdot": replace(good, files=(replace(f, url="https://huggingface.co/a/../x"),)),
            "traversal name": replace(good, files=(replace(f, name="../x.gguf"),)),
            "separator name": replace(good, files=(replace(f, name="a/b.gguf"),)),
            "device name": replace(good, files=(replace(f, name="con.gguf"),)),
            "hidden name": replace(good, files=(replace(f, name=".x"),)),
            "bad hash": replace(good, files=(replace(f, sha256="zz"),)),
            "too big": replace(good, files=(replace(f, size=models.MAX_MODEL_BYTES + 1),)),
            "too small": replace(good, files=(replace(f, size=10),)),
            "bad subdir": replace(good, subdir=".."),
            "bad id": replace(good, id="a/b"),
        }
        for label, m in bad.items():
            with self.subTest(label), self.assertRaises(ModelError):
                models.validate_model(m)

    def test_opener_rejects_unsafe_urls_before_any_request(self):
        op = cuda_pack.make_opener(models.HF_HOSTS)
        for url in ("http://huggingface.co/x", "https://huggingface.co:8443/x", "https://u@huggingface.co/x",
                    "https://evil.example/x", "https://huggingface.co.evil.example/x"):
            with self.subTest(url), self.assertRaises(cuda_pack.PackError), op(url):
                pass


    def test_observed_cdn_host_is_allowed_and_lookalikes_are_not(self):
        self.assertIn("us.aws.cdn.hf.co", models.HF_HOSTS)
        for host in ("us.aws.cdn.hf.co.evil.com", "evilus.aws.cdn.hf.co", "cdn.hf.co", "aws.cdn.hf.co", "x.us.aws.cdn.hf.co"):
            self.assertNotIn(host, models.HF_HOSTS)
            with self.assertRaises(cuda_pack.PackError):
                cuda_pack._url_ok(f"https://{host}/f", models.HF_HOSTS, False)
        self.assertEqual(cuda_pack._url_ok("https://us.aws.cdn.hf.co/f", models.HF_HOSTS, False), "https://us.aws.cdn.hf.co/f")


class FitTests(unittest.TestCase):
    def m(self, mid):
        return CATALOG[mid]

    def test_nvidia_big_card(self):
        hw = Hardware(vram_mb=32768, gpu_name="RTX 5090", ram_mb=65536, llm_runtime="cuda", speech_runtime="cuda", preset="small")
        for mid in ("llm-tiny", "llm-small", "llm-medium", "stt-large-v3-turbo"):
            self.assertEqual(models.suitability(self.m(mid), hw), ("good", "Fits your GPU"), mid)

    def test_nvidia_small_card_falls_back_to_cpu_words(self):
        hw = Hardware(vram_mb=6144, ram_mb=16384, llm_runtime="cuda", speech_runtime="cuda", preset="small")
        self.assertEqual(models.suitability(self.m("llm-small"), hw)[0], "good")  # (6144-2048)/1024 = 4 GB
        self.assertEqual(models.suitability(self.m("llm-medium"), hw), ("slow", "Runs on the CPU, slower"))

    def test_discrete_amd_uses_all_of_its_memory_for_the_llm(self):
        hw = Hardware(vram_mb=8192, ram_mb=32768, llm_runtime="vulkan", speech_runtime="cpu", preset="small")
        self.assertEqual(models.suitability(self.m("llm-small"), hw)[0], "good")
        self.assertEqual(models.suitability(self.m("llm-medium"), hw)[0], "good")
        self.assertEqual(models.suitability(self.m("stt-large-v3-turbo"), hw), ("slow", "Runs on the CPU, slower"))

    def test_integrated_or_no_gpu(self):
        hw = Hardware(ram_mb=16384, llm_runtime="cpu", speech_runtime="cpu", preset="tiny")
        self.assertEqual(models.suitability(self.m("llm-tiny"), hw), ("slow", "Runs on the CPU, slower"))
        self.assertEqual(models.suitability(self.m("stt-small.en"), hw), ("good", "Runs on the CPU"))
        self.assertEqual(models.suitability(self.m("tts-kokoro"), hw), ("good", "Runs on the CPU"))

    def test_tiny_ram_says_too_big_and_nothing_is_filtered(self):
        hw = Hardware(ram_mb=4096, llm_runtime="cpu", speech_runtime="cpu", preset="tiny")
        self.assertEqual(models.suitability(self.m("llm-medium"), hw), ("no", "Too big for this computer"))
        self.assertEqual(models.suitability(self.m("llm-tiny"), hw)[0], "slow")  # 2x2 GB = 4 GB
        verdicts = {mid: models.suitability(m, hw)[0] for mid, m in CATALOG.items()}
        self.assertEqual(set(verdicts), set(CATALOG))  # every model gets a verdict
        self.assertTrue(all(v in ("good", "slow", "no") for v in verdicts.values()))

    def test_unknown_ram_is_not_claimed_as_a_fit(self):
        hw = Hardware(ram_mb=0, llm_runtime="cpu", speech_runtime="cpu")
        level, words = models.suitability(self.m("llm-small"), hw)
        self.assertEqual(level, "slow")
        self.assertIn("Could not check", words)

    def test_recommended(self):
        gpu_hw = Hardware(llm_runtime="cuda", speech_runtime="cuda", preset="small")
        self.assertEqual(models.recommended_ids(gpu_hw), {"llm-small", "stt-large-v3-turbo", "tts-kokoro"})
        self.assertEqual(models.recommended_ids(Hardware()), {"llm-tiny", "stt-small.en", "tts-kokoro"})
        self.assertIn("stt-small", models.recommended_ids(Hardware(language="de")))

    def test_hardware_probe_returns_a_value(self):
        hw = models.detect_hardware(Path(tempfile.gettempdir()))
        self.assertIsInstance(hw, Hardware)
        self.assertGreaterEqual(hw.ram_mb, 0)


class Handler(http.server.BaseHTTPRequestHandler):
    routes: dict = {}
    seen: list = []

    def log_message(self, *a): pass

    def do_GET(self):
        Handler.seen.append(dict(self.headers))
        kind, arg = Handler.routes.get(self.path, ("404", None))
        if kind == "data":
            self.send_response(200)
            self.send_header("Content-Length", str(len(arg)))
            self.end_headers()
            self.wfile.write(arg)
        elif kind == "redirect":
            self.send_response(302)
            self.send_header("Location", arg())
            self.send_header("Content-Length", "0")
            self.end_headers()
        elif kind == "bigcl":  # announces more than it will send
            self.send_response(200)
            self.send_header("Content-Length", str(len(arg) * 3))
            self.end_headers()
            self.wfile.write(arg)
        else:
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()


class RealOpenerTests(Scratch):
    @classmethod
    def setUpClass(cls):
        cls.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()
        cls.base = f"http://127.0.0.1:{cls.server.server_port}"
        cls.foreign = f"http://localhost:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def setUp(self):
        super().setUp()
        Handler.routes, Handler.seen = {}, []

    def model(self, path="/a", data=DATA, **kw):
        return make_model({"a.gguf": data}, url=lambda n: self.base + path, hosts=frozenset({"127.0.0.1"}), **kw)

    def mgr(self, m):
        return self.manager(m, _allow_http_localhost=True)

    def test_happy_path_through_the_real_opener(self):
        Handler.routes = {"/a": ("data", DATA)}
        m = self.model()
        mgr = self.mgr(m)
        st = self.run_download(mgr, m.id)
        self.assertEqual(st.phase, "done", st.message)
        dest = self.home / "models" / "llm" / "a.gguf"
        self.assertEqual(dest.read_bytes(), DATA)
        self.assertEqual(list(dest.parent.glob("*.part")), [])
        self.assertEqual(mgr.status(m).state, "installed")

    def test_no_credentials_or_cookies_are_sent(self):
        Handler.routes = {"/a": ("data", DATA)}
        m = self.model()
        self.run_download(self.mgr(m), m.id)
        sent = {k.lower() for h in Handler.seen for k in h}
        self.assertTrue(sent <= {"host", "user-agent", "accept", "accept-encoding", "connection", "range"}, sent)

    def test_tampered_download_is_deleted_and_reported_as_a_safety_failure(self):
        Handler.routes = {"/a": ("data", DATA[:-1] + b"X")}
        m = self.model()
        st = self.run_download(self.mgr(m), m.id)
        self.assertEqual(st.phase, "error")
        self.assertIn("safety check", st.message)
        self.assertFalse(any((self.home / "models").rglob("a.gguf*")))
        self.assertNotIn("http", st.message)

    def test_content_length_over_the_pinned_size_is_refused(self):
        Handler.routes = {"/a": ("bigcl", DATA)}
        m = self.model()
        st = self.run_download(self.mgr(m), m.id)
        self.assertEqual(st.phase, "error")
        self.assertIn("safety check", st.message)
        self.assertFalse(any((self.home / "models").rglob("a.gguf*")))

    def test_redirect_to_a_host_outside_the_set_is_refused(self):
        Handler.routes = {"/a": ("redirect", lambda: self.foreign + "/b"), "/b": ("data", DATA)}
        m = self.model()
        st = self.run_download(self.mgr(m), m.id)
        self.assertEqual(st.phase, "error")
        self.assertIn("expected server", st.message)
        self.assertNotIn("localhost", st.message)
        self.assertEqual([h for h in Handler.seen if h.get("Host", "").startswith("localhost")], [])
        self.assertFalse(any((self.home / "models").rglob("a.gguf*")))

    def test_more_than_three_redirects_are_refused(self):
        for i in range(6):
            Handler.routes[f"/r{i}"] = ("redirect", lambda i=i: f"{self.base}/r{i + 1}")
        Handler.routes["/r6"] = ("data", DATA)
        m = self.model("/r0")
        st = self.run_download(self.mgr(m), m.id)
        self.assertEqual(st.phase, "error")

    def test_a_http_catalog_entry_is_refused_unless_the_test_hook_is_set(self):
        m = self.model()
        mgr = self.manager(m)  # no _allow_http_localhost
        with self.assertRaises(ModelError):
            mgr.download(m.id)
        self.assertEqual(Handler.seen, [])


class ManagerTests(Scratch):
    def test_pending_pin_model_is_refused_without_any_request(self):
        net = FakeNet()
        mgr = ModelManager(self.home, opener_factory=lambda hosts: net, free_space=lambda p: 10**12)
        for mid in models.pending():
            with self.subTest(mid), self.assertRaises(ModelError) as cm:
                mgr.download(mid)
            self.assertIn("no verified copy", str(cm.exception))
        self.assertEqual(net.requests, [])

    def test_only_catalog_ids_are_accepted(self):
        m = make_model()
        mgr = self.manager(m, FakeNet())
        for bad in ("https://evil.example/x.gguf", "../a.gguf", "C:\\x", "llm-test/../x", "", None, 5, "LLM-TEST"):
            with self.subTest(bad), self.assertRaises(ModelError):
                mgr.download(bad)
            with self.subTest(("delete", bad)), self.assertRaises(ModelError):
                mgr.delete(bad)

    def test_happy_path_progress_and_state(self):
        m = make_model({"a.gguf": DATA, "b.json": b"{}" * 1000})
        net = FakeNet()
        seen = []
        mgr = self.manager(m, net)
        orig = mgr._set
        mgr._set = lambda **kw: (seen.append(kw), orig(**kw))[1]
        mgr._factory = lambda hosts: (lambda url, headers=None: FakeNet(DATA if url.endswith("a.gguf") else b"{}" * 1000)(url, headers))
        st = self.run_download(mgr, m.id)
        self.assertEqual(st.phase, "done", st.message)
        self.assertEqual(st.done, st.total)
        dones = [kw["done"] for kw in seen if "done" in kw]
        self.assertEqual(dones, sorted(dones))
        self.assertEqual((self.home / "models" / "llm" / "a.gguf").read_bytes(), DATA)
        self.assertEqual(list((self.home / "models" / "llm").glob("*.part")), [])

    def test_tampered_short_and_oversized_downloads_fail_and_leave_nothing(self):
        m = make_model()
        for label, data in (("tampered", DATA[:-1] + b"X"), ("oversize", DATA + b"extra")):
            with self.subTest(label):
                mgr = self.manager(m, FakeNet(data))
                st = self.run_download(mgr, m.id)
                self.assertEqual(st.phase, "error")
                self.assertIn("safety check", st.message)
                self.assertFalse(any((self.home / "models").rglob("a.gguf*")), label)

    def test_a_clean_short_body_keeps_the_part_for_resume(self):
        m = make_model()
        st = self.run_download(self.manager(m, FakeNet(DATA[:-5])), m.id)
        self.assertEqual(st.phase, "error")
        self.assertIn("ended early", st.message)
        part = self.home / "models" / "llm" / "a.gguf.part"
        self.assertEqual(part.read_bytes(), DATA[:-5])
        self.assertFalse((self.home / "models" / "llm" / "a.gguf").exists())
        st = self.run_download(self.manager(m, FakeNet()), m.id)  # a healthy server finishes it
        self.assertEqual(st.phase, "done", st.message)
        # without resume (the default) a short body is still deleted
        with self.assertRaises(DownloadError):
            download_verified(FakeNet(DATA[:-5]), "u", self.home / "z" / "f.bin", sha(DATA), len(DATA))
        self.assertFalse((self.home / "z" / "f.bin.part").exists())

    def test_observer_gets_increasing_progress_ending_at_total(self):
        m = make_model()
        seen = []
        with mock.patch.object(download, "_observer", lambda name, done, total: seen.append((name, done, total))),                 mock.patch.object(models, "free_bytes", return_value=10**12):
            models.download_blocking(m, self.home, FakeNet())
        self.assertGreater(len(seen), 1)
        self.assertEqual({n for n, _, _ in seen}, {"a.gguf"})
        dones = [d for _, d, _ in seen]
        self.assertEqual(dones, sorted(dones))
        self.assertEqual(seen[-1], ("a.gguf", len(DATA), len(DATA)))

    def test_a_link_at_the_part_name_is_not_written_through(self):
        m = make_model()
        victim = self.home / "victim.bin"
        victim.write_bytes(b"precious")
        part = self.home / "models" / "llm" / "a.gguf.part"
        part.parent.mkdir(parents=True)
        try:
            os.symlink(victim, part)
        except (OSError, NotImplementedError):
            self.skipTest("cannot create file symlinks here")
        st = self.run_download(self.manager(m, FakeNet()), m.id)
        self.assertEqual(st.phase, "done", st.message)
        self.assertEqual(victim.read_bytes(), b"precious")

    def test_a_fresh_fetch_unlinks_the_part_name_before_writing(self):  # runs where symlinks can't be created
        dest = self.home / "z" / "f.bin"
        dest.parent.mkdir(parents=True)
        part = dest.with_name("f.bin.part")
        part.write_bytes(b"stale")
        calls = []
        real = Path.unlink
        with mock.patch.object(Path, "unlink", lambda p, *a, **k: (calls.append(p.name), real(p, *a, **k))[1]):
            download_verified(FakeNet(), "u", dest, sha(DATA), len(DATA))
        self.assertEqual(calls[0], "f.bin.part")
        self.assertEqual(dest.read_bytes(), DATA)

    def test_oversize_stops_as_soon_as_the_size_is_exceeded(self):
        m = make_model()
        chunks = []
        net = FakeNet(DATA + b"x" * (10 * 65536), on_chunk=chunks.append)
        self.run_download(self.manager(m, net), m.id)
        self.assertLess(len(chunks) * 65536, len(DATA) + 3 * 65536)

    def test_messages_never_contain_urls_or_bodies(self):
        m = make_model()
        for exc in (OSError("https://huggingface.co/secret?token=1"), cuda_pack.PackError("host x.example"),
                    download.DownloadError("body: <html>")):
            def opener(url, headers=None, exc=exc):
                raise exc
            st = self.run_download(self.manager(m, opener), m.id)
            self.assertEqual(st.phase, "error")
            for bad in ("http", "token", "<html>", "x.example"):
                self.assertNotIn(bad, st.message)

    def test_free_space_is_checked_before_any_request(self):
        m = make_model()
        net = FakeNet()
        mgr = self.manager(m, net, free_space=lambda p: models.FREE_MARGIN + len(DATA) - 1)
        with self.assertRaises(ModelError) as cm:
            mgr.download(m.id)
        self.assertIn("free disk space", str(cm.exception))
        self.assertEqual(net.requests, [])
        self.manager(m, net, free_space=lambda p: models.FREE_MARGIN + len(DATA)).download(m.id)

    def test_already_installed_is_refused(self):
        m = make_model()
        self.run_download(self.manager(m, FakeNet()), m.id)
        net = FakeNet()
        with self.assertRaises(ModelError):
            self.manager(m, net).download(m.id)
        self.assertEqual(net.requests, [])

    def test_only_one_download_at_a_time(self):
        m1, m2 = make_model(mid="llm-one"), make_model({"c.gguf": DATA}, mid="llm-two", probe=("c.gguf",))
        gate, started = threading.Event(), threading.Event()

        def hold(i):
            started.set()
            gate.wait(10)
        mgr = ModelManager(self.home, {m1.id: m1, m2.id: m2}, opener_factory=lambda hosts: FakeNet(on_chunk=hold),
                           free_space=lambda p: 10**12)
        mgr.download(m1.id)
        self.assertTrue(started.wait(10))
        with self.assertRaises(ModelError) as cm:
            mgr.download(m2.id)
        self.assertIn("already downloading", str(cm.exception))
        self.assertEqual(mgr.status(m1).state, "downloading")
        gate.set()
        mgr.wait(30)
        self.assertEqual(mgr.state.phase, "done")
        mgr.download(m2.id)  # free again afterwards
        mgr.wait(30)

    def test_cancel_keeps_the_part_and_a_later_run_resumes_to_the_right_hash(self):
        m = make_model()
        mgr = self.manager(m)
        net = FakeNet(on_chunk=lambda i: mgr.cancel() if i >= 3 * 65536 else None)
        mgr._factory = lambda hosts: net
        st = self.run_download(mgr, m.id)
        self.assertEqual(st.phase, "cancelled")
        part = self.home / "models" / "llm" / "a.gguf.part"
        self.assertTrue(part.exists() and 0 < part.stat().st_size < len(DATA))
        self.assertEqual(mgr.status(m).state, "partial")
        net2 = FakeNet()
        mgr2 = self.manager(m, net2)
        st = self.run_download(mgr2, m.id)
        self.assertEqual(st.phase, "done", st.message)
        self.assertIn("Range", net2.requests[0][1])
        self.assertEqual((self.home / "models" / "llm" / "a.gguf").read_bytes(), DATA)
        self.assertFalse(part.exists())

    def test_lying_servers_on_resume_still_end_verified(self):
        m = make_model()
        for mode in ("ignore-range", "bad-content-range"):
            with self.subTest(mode):
                part = self.home / "models" / "llm" / "a.gguf.part"
                part.parent.mkdir(parents=True, exist_ok=True)
                part.write_bytes(DATA[:100_000])
                net = FakeNet(mode=mode)
                st = self.run_download(self.manager(m, net), m.id)
                self.assertEqual(st.phase, "done", st.message)
                self.assertEqual((self.home / "models" / "llm" / "a.gguf").read_bytes(), DATA)
                self.assertEqual(len(net.requests), 2)  # the ranged request, then a fresh one
                (self.home / "models" / "llm" / "a.gguf").unlink()

    def test_a_corrupt_part_is_discarded_when_the_final_hash_fails(self):
        m = make_model()
        part = self.home / "models" / "llm" / "a.gguf.part"
        part.parent.mkdir(parents=True, exist_ok=True)
        part.write_bytes(b"\x00" * 100_000)  # not the start of DATA
        st = self.run_download(self.manager(m, FakeNet()), m.id)
        self.assertEqual(st.phase, "error")
        self.assertIn("safety check", st.message)
        self.assertFalse(any((self.home / "models").rglob("a.gguf*")))

    def test_scan_states(self):
        m = make_model({"a.gguf": DATA}, mid="llm-a")
        w = make_model({"model.bin": DATA}, kind="stt", mid="stt-small.en", subdir="small.en", probe=("model.bin",))
        w = replace(w, key="small.en")
        mgr = ModelManager(self.home, {m.id: m, w.id: w}, free_space=lambda p: 10**12)
        self.assertEqual({k: v.state for k, v in mgr.scan().items()}, {"llm-a": "not_installed", "stt-small.en": "not_installed"})
        llm = self.home / "models" / "llm"
        llm.mkdir(parents=True)
        (llm / "a.gguf.part").write_bytes(b"123")
        self.assertEqual(mgr.status(m), models.Status("partial", 3))
        (llm / "a.gguf").write_bytes(DATA[:-1])  # wrong size is not "installed"
        self.assertNotEqual(mgr.status(m).state, "installed")
        (llm / "a.gguf").write_bytes(DATA)
        self.assertEqual(mgr.status(m), models.Status("installed", len(DATA)))
        (self.home / "models" / "whisper" / "models--Systran--faster-whisper-small.en").mkdir(parents=True)
        self.assertEqual(mgr.status(w).state, "legacy")

    def test_dest_that_is_a_link_is_refused(self):
        m = make_model()
        target = self.home / "elsewhere"
        target.mkdir()
        llm = self.home / "models" / "llm"
        llm.parent.mkdir(parents=True)
        make_link(self, llm, target)
        net = FakeNet()
        with self.assertRaises(ModelError):
            self.manager(m, net).download(m.id)
        self.assertEqual(net.requests, [])
        self.assertEqual(list(target.iterdir()), [])


class DeleteTests(Scratch):
    def setUp(self):
        super().setUp()
        self.llm = self.home / "models" / "llm"
        self.llm.mkdir(parents=True)

    def test_delete_frees_the_files_and_the_part(self):
        m = make_model()
        (self.llm / "a.gguf").write_bytes(DATA)
        (self.llm / "a.gguf.part").write_bytes(b"12345")
        (self.llm / "other.gguf").write_bytes(b"mine")
        freed = self.manager(m).delete(m.id)
        self.assertEqual(freed, len(DATA) + 5)
        self.assertEqual([p.name for p in self.llm.iterdir()], ["other.gguf"])

    def test_in_use_model_is_refused(self):
        m = make_model()
        (self.llm / "a.gguf").write_bytes(DATA)
        with self.assertRaises(ModelError) as cm:
            self.manager(m, in_use=lambda: {m.id}).delete(m.id)
        self.assertIn("in use", str(cm.exception))
        self.assertTrue((self.llm / "a.gguf").exists())

    def test_downloading_model_is_refused(self):
        m = make_model()
        gate, started = threading.Event(), threading.Event()
        mgr = self.manager(m, FakeNet(on_chunk=lambda i: (started.set(), gate.wait(10))))
        mgr.download(m.id)
        self.assertTrue(started.wait(10))
        with self.assertRaises(ModelError) as cm:
            mgr.delete(m.id)
        self.assertIn("downloading", str(cm.exception))
        gate.set()
        mgr.wait(30)

    def test_own_file_and_unknown_ids_are_never_deletable(self):
        own = self.llm / "my-own-model.gguf"
        own.write_bytes(b"mine")
        mgr = self.manager(make_model())
        for bad in ("my-own-model.gguf", str(own), "Your own file", "../llm/my-own-model.gguf"):
            with self.subTest(bad), self.assertRaises(ModelError):
                mgr.delete(bad)
        self.assertTrue(own.exists())

    def test_own_llm_file_is_reported_read_only(self):
        own = self.llm / "my-own-model.gguf"
        own.write_bytes(b"mine")
        cfg = Config()
        cfg.llm.model_path = str(own)
        found = models.own_llm_file(cfg, self.home)
        self.assertEqual((found.label, found.path, found.bytes), ("Your own file", own, 4))
        self.assertNotIn(found.label, CATALOG)
        self.assertIsNone(models.own_llm_file(Config(), self.home))

    def test_a_target_outside_models_is_refused(self):
        outside = self.home / "keep"
        outside.mkdir()
        (outside / "a.gguf").write_bytes(b"precious")
        m = replace(make_model(), subdir="../../keep")
        with self.assertRaises(ModelError):
            self.manager(m).delete(m.id)
        self.assertTrue((outside / "a.gguf").exists())

    def test_a_link_inside_models_is_not_followed(self):
        outside = self.home / "keep"
        outside.mkdir()
        (outside / "x.bin").write_bytes(b"precious")
        w = make_model({"model.bin": DATA}, kind="stt", mid="stt-t", subdir="t", probe=("model.bin",))
        (self.home / "models" / "whisper").mkdir()
        link = self.home / "models" / "whisper" / "t"
        make_link(self, link, outside)
        with self.assertRaises(ModelError):
            self.manager(w).delete(w.id)
        self.assertTrue((outside / "x.bin").exists())

    def test_a_whisper_folder_is_removed_as_a_whole(self):
        w = make_model({"model.bin": DATA}, kind="stt", mid="stt-t", subdir="t", probe=("model.bin",))
        d = self.home / "models" / "whisper" / "t"
        d.mkdir(parents=True)
        (d / "model.bin").write_bytes(DATA)
        (d / "config.json").write_bytes(b"{}")
        other = self.home / "models" / "whisper" / "u"
        other.mkdir()
        self.assertEqual(self.manager(w).delete(w.id), len(DATA) + 2)
        self.assertFalse(d.exists())
        self.assertTrue(other.exists())

    def test_shared_kokoro_voices_file_stays_while_another_model_needs_it(self):
        a = make_model({"a.onnx": DATA, "voices.bin": b"v" * 2_000_000}, kind="tts", mid="tts-a", probe=("a.onnx",))
        b = make_model({"b.onnx": DATA + b"1", "voices.bin": b"v" * 2_000_000}, kind="tts", mid="tts-b", probe=("b.onnx",))
        k = self.home / "models" / "kokoro"
        k.mkdir(parents=True)
        for n, data in (("a.onnx", DATA), ("b.onnx", DATA + b"1"), ("voices.bin", b"v" * 2_000_000)):
            (k / n).write_bytes(data)
        mgr = ModelManager(self.home, {a.id: a, b.id: b})
        mgr.delete("tts-a")
        self.assertEqual(sorted(p.name for p in k.iterdir()), ["b.onnx", "voices.bin"])
        mgr.delete("tts-b")
        self.assertEqual(list(k.iterdir()), [])

    def test_a_locked_file_gives_a_short_message(self):
        m = make_model()
        (self.llm / "a.gguf").write_bytes(DATA)
        with mock.patch.object(Path, "unlink", side_effect=PermissionError(13, "in use")):
            with self.assertRaises(ModelError) as cm:
                self.manager(m).delete(m.id)
        self.assertIn("open in another program", str(cm.exception))


class InUseTests(Scratch):
    def test_in_use_ids_returns_the_callback_result_as_a_copy(self):
        ids = {"llm-tiny", "tts-kokoro"}
        mgr = ModelManager(self.home, in_use=lambda: ids)
        got = mgr.in_use_ids()
        self.assertEqual(got, {"llm-tiny", "tts-kokoro"})
        got.add("x")
        self.assertEqual(ids, {"llm-tiny", "tts-kokoro"})

    def test_ids_in_use_from_config(self):
        cfg = Config()
        cfg.llm.preset = "small"
        used = models.ids_in_use(cfg, self.home, "small.en")
        self.assertEqual(used, {"llm-small", "stt-small.en", "tts-kokoro"})
        cfg.llm.model_path = str(self.home / "mine.gguf")
        self.assertEqual(models.ids_in_use(cfg, self.home, "tiny.en"), {"stt-tiny.en", "tts-kokoro"})


class LlamaServerTests(Scratch):
    def cfg(self, preset):
        cfg = Config()
        cfg.llm.preset = preset
        return cfg.llm

    def test_resolve_model_never_downloads(self):
        with mock.patch.object(llama_server, "download") as dl:
            path, repo, file = llama_server.resolve_model(self.cfg("medium"), self.home)
        dl.assert_not_called()
        self.assertEqual((path, repo, file), (self.home / "models" / "llm" / "Qwen3.5-9B-Q4_K_M.gguf",
                                              "unsloth/Qwen3.5-9B-GGUF", "Qwen3.5-9B-Q4_K_M.gguf"))
        cfg = Config()
        cfg.llm.model_path = str(self.home / "x.gguf")
        self.assertEqual(llama_server.resolve_model(cfg.llm, self.home), (self.home / "x.gguf", None, None))

    def test_unpinned_medium_is_refused_and_old_presets_keep_working(self):
        unpinned = {k: replace(CATALOG[k], files=()) for k in ("llm-small", "llm-medium")}  # whatever model_pins holds today
        with mock.patch.dict(models.CATALOG, unpinned), mock.patch.object(llama_server, "download") as dl:
            with self.assertRaises(SystemExit):
                llama_server.ensure_model(self.cfg("medium"), self.home)
            dl.assert_not_called()
            llama_server.ensure_model(self.cfg("small"), self.home)
            dl.assert_called_once()

    def test_a_pinned_preset_goes_through_the_verified_manager(self):
        m = make_model({"Qwen3.5-2B-Q4_K_M.gguf": DATA}, mid="llm-tiny", probe=("Qwen3.5-2B-Q4_K_M.gguf",))
        with mock.patch.dict(models.CATALOG, {"llm-tiny": m}), \
                mock.patch.object(models, "free_bytes", return_value=10**12), \
                mock.patch.object(llama_server, "download") as dl:
            real = models.ModelManager

            def mgr(home, cat, **kw):
                return real(home, cat, **{**kw, "opener_factory": lambda hosts: FakeNet()})
            with mock.patch.object(models, "ModelManager", mgr):
                path = llama_server.ensure_model(self.cfg("tiny"), self.home)
        dl.assert_not_called()
        self.assertEqual(path.read_bytes(), DATA)

    def test_a_custom_repo_stays_unpinned(self):
        cfg = Config()
        cfg.llm.model_repo, cfg.llm.model_file = "someone/model", "m.gguf"
        with mock.patch.object(llama_server, "download") as dl:
            llama_server.ensure_model(cfg.llm, self.home)
        self.assertIn("/someone/model/resolve/main/m.gguf", dl.call_args[0][0])


class DownloadVerifiedResume(Scratch):
    def dest(self):
        return self.home / "d" / "f.bin"

    def test_default_still_starts_fresh_and_deletes_the_part_on_cancel(self):
        ev = threading.Event()
        ev.set()
        with self.assertRaises(DownloadCancelled):
            download_verified(FakeNet(), "u", self.dest(), sha(DATA), len(DATA), cancel=ev)
        self.assertFalse(self.dest().with_name("f.bin.part").exists())

    def test_old_style_opener_without_headers_still_works(self):
        from contextlib import contextmanager as cm

        @cm
        def op(url):
            yield Resp(DATA)
        download_verified(op, "u", self.dest(), sha(DATA), len(DATA))
        self.assertEqual(self.dest().read_bytes(), DATA)

    def test_resume_requires_a_matching_content_range(self):
        part = self.dest().with_name("f.bin.part")
        part.parent.mkdir(parents=True)
        part.write_bytes(DATA[:1000])
        net = FakeNet()
        download_verified(net, "u", self.dest(), sha(DATA), len(DATA), resume=True)
        self.assertEqual(net.requests[0][1], {"Range": "bytes=1000-"})
        self.assertEqual(self.dest().read_bytes(), DATA)

    def test_content_length_over_the_size_is_refused(self):
        class Big(FakeNet):
            def __call__(self, url, headers=None):
                self.requests.append((url, headers))

                @contextmanager
                def ctx():
                    yield Resp(DATA, 200, {"content-length": str(len(DATA) + 1)})
                return ctx()
        with self.assertRaises(DownloadError):
            download_verified(Big(), "u", self.dest(), sha(DATA), len(DATA))
        self.assertFalse(self.dest().exists())


if __name__ == "__main__":
    unittest.main()
