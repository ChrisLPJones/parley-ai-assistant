"""CUDA pack: generator vs uv.lock, manifest validation, verified download, safe extraction, atomic install, gating.

Local http.server fixtures with tiny fake wheels; every Parley home is a temp dir `parley_test_cuda_pack_*`.
The real ~1.6 GB download is checked by hand only (see PROJECT.md "Tests").

Run: uv run python -m unittest discover -s tests -v
"""
import importlib.util
import io
import json
import os
import shutil
import sys
import tempfile
import tomllib
import unittest
import zipfile
from pathlib import Path
from unittest import mock

from parley import cuda_pack, gpu
from parley.config import Config
from parley.download import DownloadError
from test_update import FakeServer, sha

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("parley_cuda_pack_script", ROOT / "scripts" / "cuda_pack.py")
gen = importlib.util.module_from_spec(_spec)
sys.modules["parley_cuda_pack_script"] = gen
_spec.loader.exec_module(gen)

CUDA_PLAN = gpu.Plan("cuda", "cuda", "cuda", "small", "RTX")
CPU_PLAN = gpu.Plan("cpu", "cpu", "cpu", "tiny", "none")
HOSTS = {"127.0.0.1"}


def wheel_bytes(members: dict[str, bytes], extra: list[tuple[str, bytes]] = ()) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in {**members}.items():
            zf.writestr(name, data)
        with warnings_off():
            for name, data in extra:
                zf.writestr(name, data)
    return buf.getvalue()


class warnings_off:
    def __enter__(self):
        import warnings
        self._c = warnings.catch_warnings()
        self._c.__enter__()
        warnings.simplefilter("ignore")

    def __exit__(self, *a):
        self._c.__exit__(*a)


def good_wheels() -> list[tuple[str, dict[str, bytes]]]:
    return [("nvidia-cublas-cu12", {"nvidia/cublas/bin/cublas64_12.dll": b"CUBLAS" * 100,
                                    "nvidia/cublas/bin/cublasLt64_12.dll": b"LT" * 300,
                                    "nvidia/cublas/include/cublas.h": b"header"}),
            ("nvidia-cuda-runtime-cu12", {"nvidia/cuda_runtime/bin/cudart64_12.dll": b"RT" * 50})]


class Fixture(unittest.TestCase):
    def setUp(self):
        self.srv = FakeServer()
        self.addCleanup(self.srv.close)
        self.tmp = tempfile.TemporaryDirectory(prefix="parley_test_cuda_pack_")
        self.addCleanup(self.tmp.cleanup)
        self.home = Path(self.tmp.name)
        self.opener = cuda_pack.make_opener(HOSTS, _allow_http_localhost=True)

    def manifest(self, wheels=None, tamper: set[str] = frozenset(), version: str = "1") -> cuda_pack.Manifest:
        """Publish fake wheels on the fixture and return the manifest for them. `tamper`: serve other bytes of the
        same size under the hash that was announced."""
        items = []
        for pkg, members in wheels or good_wheels():
            data = wheel_bytes(members)
            path = f"/{pkg}-{version}-py3-none-win_amd64.whl"
            self.srv.routes[path] = (b"X" * len(data)) if pkg in tamper else data
            items.append({"package": pkg, "url": self.srv.base + path, "sha256": sha(data), "size": len(data),
                          "dlls": [n for n in members if n.endswith(".dll")]})
        doc = {"schema": 1, "platform": "win_amd64", "wheels": items, "version": version}
        return cuda_pack.parse_manifest(json.dumps(doc), HOSTS, _allow_http_localhost=True)

    def cfg(self, **kw) -> Config:
        c = Config()
        for k, v in kw.items():
            sect, key = k.split("__")
            setattr(getattr(c, sect), key, v)
        return c

    def assertNoPack(self):
        runtime = self.home / "runtime"
        self.assertFalse((runtime / "cuda12").exists())
        self.assertFalse((runtime / "cuda12.partial").exists())


class Generator(unittest.TestCase):
    lock = tomllib.loads((ROOT / "uv.lock").read_text("utf-8"))

    def test_committed_manifest_matches_the_lock(self):
        text = gen.render(gen.build_manifest((ROOT / "uv.lock").read_text("utf-8")))
        self.assertEqual((ROOT / "src" / "parley" / "cuda_pack.json").read_text("utf-8"), text,
                         "run: uv run python scripts/cuda_pack.py")

    def test_urls_hashes_and_sizes_come_from_the_lock_exactly(self):
        m = json.loads((ROOT / "src" / "parley" / "cuda_pack.json").read_text("utf-8"))
        pkgs = {p["name"]: p for p in self.lock["package"]}
        self.assertEqual({w["package"] for w in m["wheels"]}, {n for n in pkgs if n.startswith("nvidia-") and n.endswith("-cu12")})
        for w in m["wheels"]:
            hits = [x for x in pkgs[w["package"]]["wheels"] if x["url"] == w["url"]]
            self.assertEqual(len(hits), 1)
            self.assertTrue(w["url"].endswith("win_amd64.whl"))
            self.assertEqual("sha256:" + w["sha256"], hits[0]["hash"])
            self.assertEqual(w["size"], hits[0]["size"])
            self.assertEqual(w["version"], pkgs[w["package"]]["version"])

    def test_bundled_manifest_passes_runtime_validation(self):
        m = cuda_pack.load_manifest()
        self.assertIsNotNone(m)
        self.assertEqual(len(m.wheels), 6)
        self.assertTrue({urlpart for w in m.wheels for urlpart in [w.url.split("/")[2]]} <= cuda_pack.ALLOWED_HOSTS)
        self.assertEqual(set(gen.ALLOWED_HOSTS), set(cuda_pack.ALLOWED_HOSTS))

    def test_listed_dlls_exist_in_the_installed_wheels(self):
        try:
            import nvidia
        except ImportError:
            self.skipTest("nvidia-* wheels are not installed")
        base = Path(list(nvidia.__path__)[0])
        for w in cuda_pack.load_manifest().wheels:
            for d in w.dlls:
                self.assertTrue((base.parent / d).is_file(), d)

    def test_unknown_or_missing_nvidia_package_stops_the_generator(self):
        text = (ROOT / "uv.lock").read_text("utf-8")
        extra = text + '\n[[package]]\nname = "nvidia-curand-cu12"\nversion = "1"\nsource = { registry = "x" }\n'
        with self.assertRaises(gen.PackError):
            gen.build_manifest(extra)
        with self.assertRaises(gen.PackError):
            gen.build_manifest(text.replace('name = "nvidia-cufft-cu12"', 'name = "other-cufft"'))

    def test_foreign_host_in_the_lock_is_refused(self):
        text = (ROOT / "uv.lock").read_text("utf-8").replace("https://files.pythonhosted.org/packages/20/e2",
                                                             "https://evil.example/packages/20/e2")
        with self.assertRaises(gen.PackError):
            gen.build_manifest(text)

    def test_generate_writes_and_check_detects_drift(self):
        with tempfile.TemporaryDirectory(prefix="parley_test_cuda_pack_gen_") as d:
            out = Path(d) / "cuda_pack.json"
            self.assertTrue(gen.generate(ROOT / "uv.lock", out))
            self.assertFalse(gen.generate(ROOT / "uv.lock", out))
            self.assertEqual(out.read_text("utf-8"), (ROOT / "src" / "parley" / "cuda_pack.json").read_text("utf-8"))


class ManifestRules(unittest.TestCase):
    def doc(self, **over):
        w = {"package": "p", "url": "https://files.pythonhosted.org/a/p.whl", "sha256": "a" * 64, "size": 10,
             "dlls": ["nvidia/p/bin/p64.dll"]}
        w.update(over)
        return json.dumps({"schema": 1, "platform": "win_amd64", "wheels": [w]})

    def test_good(self):
        self.assertEqual(cuda_pack.parse_manifest(self.doc()).files, {"p64.dll"})

    def test_bad_members_and_fields(self):
        bad_dlls = ["../evil.dll", "nvidia/p/bin/../../evil.dll", "/abs/p.dll", "C:/p.dll", "nvidia\\p\\bin\\p.dll",
                    "nvidia/p/bin/p.exe", "nvidia/p/lib/p.dll", "nvidia/p/bin/con.dll", "nvidia/p/bin/sub/p.dll",
                    "nvidia/p/bin/p .dll", "nvidia/p/bin/p\u202e.dll", "nvidia/p/bin/.dll", "nvidia/P/bin/p.dll"]
        for d in bad_dlls:
            with self.subTest(d=d), self.assertRaises(cuda_pack.PackError):
                cuda_pack.parse_manifest(self.doc(dlls=[d]))
        for over in ({"sha256": "A" * 64}, {"sha256": "a" * 63}, {"size": 0}, {"size": cuda_pack.MAX_WHEEL_BYTES + 1},
                     {"size": True}, {"dlls": []}, {"dlls": "x"},
                     {"dlls": ["nvidia/p/bin/a.dll", "nvidia/q/bin/A.DLL"]}):
            with self.subTest(over=over), self.assertRaises(cuda_pack.PackError):
                cuda_pack.parse_manifest(self.doc(**over))

    def test_urls(self):
        for url in ("http://files.pythonhosted.org/a.whl", "https://evil.example/a.whl", "ftp://files.pythonhosted.org/a",
                    "https://user:pw@files.pythonhosted.org/a.whl", "https://files.pythonhosted.org.evil.example/a",
                    "https://files.pythonhosted.org:bad/a", "http://127.0.0.1/a.whl", ""):
            with self.subTest(url=url), self.assertRaises(cuda_pack.PackError):
                cuda_pack.parse_manifest(self.doc(url=url))

    def test_url_length_limit_is_2048(self):
        hosts = {"files.pythonhosted.org"}
        base = "https://files.pythonhosted.org/"
        long_ok = base + "a" * (2048 - len(base))  # real signed redirect URLs are ~950 chars
        self.assertEqual(cuda_pack._url_ok(base + "a" * 950, hosts, False), base + "a" * 950)
        self.assertEqual(cuda_pack._url_ok(long_ok, hosts, False), long_ok)
        with self.assertRaises(cuda_pack.PackError):
            cuda_pack._url_ok(long_ok + "a", hosts, False)

    def test_not_json_and_wrong_schema(self):
        for raw in ("{", "[]", json.dumps({"schema": 2, "platform": "win_amd64", "wheels": []}), json.dumps({"schema": 1})):
            with self.subTest(raw=raw), self.assertRaises(cuda_pack.PackError):
                cuda_pack.parse_manifest(raw)


class Install(Fixture):
    def test_downloads_extracts_only_listed_dlls_and_stamps(self):
        m = self.manifest()
        bindir = cuda_pack.install(m, self.home, self.opener)
        self.assertEqual(bindir, self.home / "runtime" / "cuda12" / "bin")
        self.assertEqual(sorted(p.name for p in bindir.iterdir()), ["cublas64_12.dll", "cublasLt64_12.dll", "cudart64_12.dll"])
        self.assertEqual((bindir / "cudart64_12.dll").read_bytes(), b"RT" * 50)
        self.assertEqual(cuda_pack.usable_dir(self.home, m), bindir)
        pack = self.home / "runtime" / "cuda12"
        self.assertEqual(sorted(p.name for p in pack.iterdir()), ["bin", "stamp.json"])  # no .dl, no headers
        self.assertFalse((self.home / "runtime" / "cuda12.partial").exists())

    def test_hash_mismatch_is_deleted_and_nothing_is_extracted(self):
        m = self.manifest(tamper={"nvidia-cuda-runtime-cu12"})
        with mock.patch.object(cuda_pack, "extract_dlls", wraps=cuda_pack.extract_dlls) as ex:
            with self.assertRaises(DownloadError):
                cuda_pack.install(m, self.home, self.opener)
        self.assertEqual(ex.call_count, 1)  # the good first wheel only; the tampered one was never opened
        self.assertNoPack()
        self.assertIsNone(cuda_pack.usable_dir(self.home, m))

    def test_first_wheel_tampered_never_extracts(self):
        m = self.manifest(tamper={"nvidia-cublas-cu12"})
        with mock.patch.object(cuda_pack, "extract_dlls") as ex, self.assertRaises(DownloadError):
            cuda_pack.install(m, self.home, self.opener)
        ex.assert_not_called()
        self.assertNoPack()
        self.assertEqual(list((self.home / "runtime").glob("**/*.whl*")), [])

    def test_truncated_and_oversize_downloads_rejected(self):
        m = self.manifest()
        self.srv.truncate.add("/nvidia-cublas-cu12-1-py3-none-win_amd64.whl")  # announces more than it sends
        with self.assertRaises(Exception):
            cuda_pack.install(m, self.home, self.opener)
        self.assertNoPack()
        self.srv.truncate.clear()
        self.srv.routes["/nvidia-cublas-cu12-1-py3-none-win_amd64.whl"] += b"extra"
        with self.assertRaises(DownloadError):
            cuda_pack.install(m, self.home, self.opener)
        self.assertNoPack()

    def test_zip_with_odd_unlisted_members_ignores_them(self):
        members = {"nvidia/p/bin/p64.dll": b"ok", "../evil.dll": b"x", "/abs.dll": b"x", "C:/evil.dll": b"x",
                   "nvidia\\p\\bin\\q.dll": b"x", "nvidia/p/bin/other.dll": b"x"}
        data = wheel_bytes(members)
        self.srv.routes["/p.whl"] = data
        doc = {"schema": 1, "platform": "win_amd64", "wheels": [
            {"package": "p", "url": self.srv.base + "/p.whl", "sha256": sha(data), "size": len(data),
             "dlls": ["nvidia/p/bin/p64.dll"]}]}
        m = cuda_pack.parse_manifest(json.dumps(doc), HOSTS, _allow_http_localhost=True)
        cuda_pack.install(m, self.home, self.opener)
        self.assertEqual(sorted(p.name for p in (self.home / "runtime" / "cuda12" / "bin").iterdir()), ["p64.dll"])
        self.assertEqual(sorted(p.name for p in (self.home / "runtime").iterdir()), ["cuda12", "cuda12.lock"])

    def test_listed_member_missing_duplicated_or_directory_rejected(self):
        for label, data in (("missing", wheel_bytes({"nvidia/p/bin/other.dll": b"x"})),
                            ("duplicate", wheel_bytes({"nvidia/p/bin/p64.dll": b"a"}, [("nvidia/p/bin/p64.dll", b"b")])),
                            ("directory", wheel_bytes({"nvidia/p/bin/p64.dll/": b""}))):
            with self.subTest(label), tempfile.TemporaryDirectory(prefix="parley_test_cuda_pack_w_") as d:
                self.srv.routes["/p.whl"] = data
                doc = {"schema": 1, "platform": "win_amd64", "wheels": [
                    {"package": "p", "url": self.srv.base + "/p.whl", "sha256": sha(data), "size": len(data),
                     "dlls": ["nvidia/p/bin/p64.dll"]}]}
                m = cuda_pack.parse_manifest(json.dumps(doc), HOSTS, _allow_http_localhost=True)
                with self.assertRaises(cuda_pack.PackError):
                    cuda_pack.install(m, Path(d), self.opener)
                self.assertFalse((Path(d) / "runtime" / "cuda12").exists())
                self.assertFalse((Path(d) / "runtime" / "cuda12.partial").exists())

    def test_oversize_dll_rejected(self):
        m = self.manifest()
        with mock.patch.object(cuda_pack, "MAX_DLL_BYTES", 100), self.assertRaises(cuda_pack.PackError):
            cuda_pack.install(m, self.home, self.opener)
        self.assertNoPack()

    def test_extraction_never_overwrites(self):
        m = self.manifest([("p", {"nvidia/p/bin/p64.dll": b"new"})])
        bin_dir = self.home / "b"
        bin_dir.mkdir()
        (bin_dir / "p64.dll").write_bytes(b"old")
        wheel = self.home / "w.whl"
        wheel.write_bytes(wheel_bytes({"nvidia/p/bin/p64.dll": b"new"}))
        with self.assertRaises(FileExistsError):
            cuda_pack.extract_dlls(wheel, m.wheels[0], bin_dir)
        self.assertEqual((bin_dir / "p64.dll").read_bytes(), b"old")

    def test_failure_on_second_wheel_leaves_no_pack(self):
        m = self.manifest()
        del self.srv.routes["/nvidia-cuda-runtime-cu12-1-py3-none-win_amd64.whl"]
        with self.assertRaises(Exception):
            cuda_pack.install(m, self.home, self.opener)
        self.assertNoPack()
        self.assertIsNone(cuda_pack.usable_dir(self.home, m))

    def test_interrupt_mid_way_leaves_no_usable_pack(self):
        m = self.manifest()
        calls = []

        def killer(url):
            calls.append(url)
            if len(calls) == 2:
                raise KeyboardInterrupt  # the process is stopped between wheels
            return self.opener(url)

        with self.assertRaises(KeyboardInterrupt):
            cuda_pack.install(m, self.home, killer)
        self.assertNoPack()

    def test_crash_leftovers_are_not_a_pack_and_are_cleaned_next_time(self):
        m = self.manifest()
        partial = self.home / "runtime" / "cuda12.partial" / "bin"
        partial.mkdir(parents=True)
        (partial / "cudart64_12.dll").write_bytes(b"half")
        self.assertIsNone(cuda_pack.usable_dir(self.home, m))
        cuda_pack.install(m, self.home, self.opener)
        self.assertEqual((self.home / "runtime" / "cuda12" / "bin" / "cudart64_12.dll").read_bytes(), b"RT" * 50)
        self.assertFalse((self.home / "runtime" / "cuda12.partial").exists())

    def test_pack_without_stamp_or_with_changed_file_is_not_usable(self):
        m = self.manifest()
        cuda_pack.install(m, self.home, self.opener)
        pack = self.home / "runtime" / "cuda12"
        dll = pack / "bin" / "cudart64_12.dll"
        dll.write_bytes(b"short")
        self.assertIsNone(cuda_pack.usable_dir(self.home, m))
        dll.write_bytes(b"RT" * 50)
        self.assertIsNotNone(cuda_pack.usable_dir(self.home, m))
        dll.unlink()
        self.assertIsNone(cuda_pack.usable_dir(self.home, m))
        cuda_pack.install(m, self.home, self.opener)
        (pack / "stamp.json").unlink()
        self.assertIsNone(cuda_pack.usable_dir(self.home, m))

    def test_stamp_mismatch_triggers_a_new_download_that_replaces_the_pack(self):
        m1 = self.manifest(version="1")
        cuda_pack.install(m1, self.home, self.opener)
        m2 = self.manifest([("nvidia-cuda-runtime-cu12", {"nvidia/cuda_runtime/bin/cudart64_12.dll": b"NEWRT"})], version="2")
        self.assertNotEqual(m1.stamp, m2.stamp)
        self.assertIsNone(cuda_pack.usable_dir(self.home, m2))
        self.assertTrue(cuda_pack.maybe_download(self.cfg(), CUDA_PLAN, self.home, frozen=True, manifest=m2,
                                                 open_stream=self.opener))
        self.assertEqual(cuda_pack.usable_dir(self.home, m2), self.home / "runtime" / "cuda12" / "bin")
        self.assertEqual([p.name for p in (self.home / "runtime" / "cuda12" / "bin").iterdir()], ["cudart64_12.dll"])
        self.assertIsNone(cuda_pack.usable_dir(self.home, m1))
        self.assertFalse((self.home / "runtime" / "cuda12.old").exists())

    def test_failed_replacement_keeps_the_old_pack(self):
        m1 = self.manifest(version="1")
        cuda_pack.install(m1, self.home, self.opener)
        m2 = self.manifest(version="2", tamper={"nvidia-cublas-cu12"})
        with self.assertRaises(DownloadError):
            cuda_pack.install(m2, self.home, self.opener)
        self.assertIsNotNone(cuda_pack.usable_dir(self.home, m1))

    def test_second_process_is_refused_while_the_lock_is_held(self):
        m = self.manifest()
        with cuda_pack._lock(self.home / "runtime" / "cuda12.lock"):
            with self.assertRaises(cuda_pack.PackError):
                cuda_pack.install(m, self.home, self.opener)
        self.assertNoPack()
        cuda_pack.install(m, self.home, self.opener)  # released

    def test_not_enough_disk_space(self):
        m = self.manifest()
        with mock.patch("shutil.disk_usage", return_value=shutil._ntuple_diskusage(10, 9, 1)):
            with self.assertRaises(cuda_pack.PackError):
                cuda_pack.install(m, self.home, self.opener)
        self.assertNoPack()


class Gating(Fixture):
    def never_opener(self, url):
        raise AssertionError("must not touch the network")

    def run_gate(self, cfg, plan, frozen=True, manifest=None, say=None):
        return cuda_pack.maybe_download(cfg, plan, self.home, say, frozen=frozen, manifest=manifest or self.manifest(),
                                        open_stream=self.never_opener)

    def test_does_nothing_when_it_should_not(self):
        cases = {
            "not frozen": (self.cfg(), CUDA_PLAN, False),
            "no NVIDIA (plan is cpu)": (self.cfg(), CPU_PLAN, True),
            "both forced to cpu": (self.cfg(stt__device="cpu", tts__device="cpu"), CUDA_PLAN, True),
            "disabled": (self.cfg(gpu__download_cuda=False), CUDA_PLAN, True),
            "forced cuda but no NVIDIA": (self.cfg(stt__device="cuda", tts__device="cuda"), CPU_PLAN, True),
        }
        for label, (cfg, plan, frozen) in cases.items():
            with self.subTest(label):
                self.assertFalse(self.run_gate(cfg, plan, frozen))
                self.assertFalse((self.home / "runtime").exists())
                self.assertIsNone(cuda_pack.start_background(cfg, plan, self.home))

    def test_one_stage_on_cuda_is_enough(self):
        self.assertTrue(cuda_pack.needed(self.cfg(stt__device="cpu"), CUDA_PLAN))
        self.assertTrue(cuda_pack.needed(self.cfg(tts__device="cpu"), CUDA_PLAN))
        self.assertFalse(cuda_pack.needed(self.cfg(llm__runtime="cpu", stt__device="cpu", tts__device="cpu"), CUDA_PLAN))

    def test_already_installed_does_nothing(self):
        m = self.manifest()
        cuda_pack.install(m, self.home, self.opener)
        self.assertFalse(self.run_gate(self.cfg(), CUDA_PLAN, manifest=m))

    def test_downloads_in_the_background_and_speaks(self):
        said = []
        m = self.manifest()
        with mock.patch.object(cuda_pack, "is_frozen", return_value=True), \
                mock.patch.object(cuda_pack, "load_manifest", return_value=m), \
                mock.patch.object(cuda_pack, "make_opener", return_value=self.opener):
            t = cuda_pack.start_background(self.cfg(), CUDA_PLAN, self.home, said.append)
            self.assertIsNotNone(t)
            t.join(30)
        self.assertIsNotNone(cuda_pack.usable_dir(self.home, m))
        self.assertEqual(len(said), 2)
        self.assertTrue(said[0].startswith("Downloading GPU support, about "))
        self.assertIn("Parley will use the CPU until it is done.", said[0])
        self.assertIn("next time", said[1])

    def test_on_installed_runs_after_a_fresh_install_only(self):
        m, said, hook = self.manifest(), [], []
        self.assertTrue(cuda_pack.maybe_download(self.cfg(), CUDA_PLAN, self.home, said.append, frozen=True, manifest=m,
                                                 open_stream=self.opener, on_installed=lambda: hook.append(1)))
        self.assertEqual(hook, [1])
        self.assertIn("restart", said[-1])
        self.assertNotIn("next time", said[-1])
        self.assertFalse(cuda_pack.maybe_download(self.cfg(), CUDA_PLAN, self.home, frozen=True, manifest=m,
                                                  open_stream=self.never_opener, on_installed=lambda: hook.append(2)))
        self.assertEqual(hook, [1])  # already installed at startup: no restart

    def test_on_installed_not_called_on_failure_or_when_disabled(self):
        hook = []
        with self.assertLogs("parley.cuda_pack", "WARNING"):
            cuda_pack.maybe_download(self.cfg(), CUDA_PLAN, self.home, frozen=True,
                                     manifest=self.manifest(tamper={"nvidia-cublas-cu12"}), open_stream=self.opener,
                                     on_installed=lambda: hook.append(1))
        cuda_pack.maybe_download(self.cfg(gpu__download_cuda=False), CUDA_PLAN, self.home, frozen=True,
                                 manifest=self.manifest(), open_stream=self.never_opener, on_installed=lambda: hook.append(1))
        self.assertEqual(hook, [])

    def test_a_failing_hook_does_not_break_the_download(self):
        def boom():
            raise RuntimeError("x")
        with self.assertLogs("parley.cuda_pack", "ERROR"):
            self.assertTrue(cuda_pack.maybe_download(self.cfg(), CUDA_PLAN, self.home, frozen=True,
                                                     manifest=self.manifest(), open_stream=self.opener, on_installed=boom))

    def test_failures_are_quiet_and_never_raise(self):
        m = self.manifest(tamper={"nvidia-cublas-cu12"})
        said = []
        with self.assertLogs("parley.cuda_pack", "WARNING") as logs:
            self.assertFalse(cuda_pack.maybe_download(self.cfg(), CUDA_PLAN, self.home, said.append, frozen=True,
                                                      manifest=m, open_stream=self.opener))
        self.assertIn("stays on the CPU", logs.output[0])
        self.assertEqual(len(said), 1)  # only the "downloading" notice
        self.assertNoPack()
        # offline: nothing is listening on the announced port
        self.srv.close()
        with self.assertLogs("parley.cuda_pack", "WARNING"):
            self.assertFalse(cuda_pack.maybe_download(self.cfg(), CUDA_PLAN, self.home, frozen=True, manifest=m,
                                                      open_stream=self.opener))
        self.assertNoPack()

    def test_missing_manifest_is_quiet(self):
        with mock.patch.object(cuda_pack, "load_manifest", return_value=None):
            self.assertFalse(cuda_pack.maybe_download(self.cfg(), CUDA_PLAN, self.home, frozen=True))
        self.assertFalse((self.home / "runtime").exists())

    def test_size_words(self):
        self.assertEqual(cuda_pack._size_words(1_612_000_000), "1.6 gigabytes")
        self.assertEqual(cuda_pack._size_words(350_000_000), "4 hundred megabytes")


class HostAllowlist(Fixture):
    def test_redirect_to_another_host_is_refused(self):
        data = wheel_bytes({"nvidia/p/bin/p64.dll": b"x"})
        other = FakeServer()
        self.addCleanup(other.close)
        other.routes["/p.whl"] = data
        self.srv.routes["/p.whl"] = (302, {"Location": other.base.replace("127.0.0.1", "localhost") + "/p.whl"}, b"")
        doc = {"schema": 1, "platform": "win_amd64", "wheels": [
            {"package": "p", "url": self.srv.base + "/p.whl", "sha256": sha(data), "size": len(data),
             "dlls": ["nvidia/p/bin/p64.dll"]}]}
        m = cuda_pack.parse_manifest(json.dumps(doc), HOSTS, _allow_http_localhost=True)
        with self.assertRaises(cuda_pack.PackError) as cm:
            cuda_pack.install(m, self.home, self.opener)
        self.assertIn("host not allowed", str(cm.exception))
        self.assertNoPack()

    def test_redirect_on_the_same_host_is_followed_but_loops_are_not(self):
        data = wheel_bytes({"nvidia/p/bin/p64.dll": b"x"})
        self.srv.routes["/real.whl"] = data
        self.srv.routes["/p.whl"] = (302, {"Location": "/real.whl"}, b"")
        self.srv.routes["/loop.whl"] = (302, {"Location": "/loop.whl"}, b"")
        for path, ok in (("/p.whl", True), ("/loop.whl", False)):
            with self.subTest(path), tempfile.TemporaryDirectory(prefix="parley_test_cuda_pack_r_") as d:
                doc = {"schema": 1, "platform": "win_amd64", "wheels": [
                    {"package": "p", "url": self.srv.base + path, "sha256": sha(data), "size": len(data),
                     "dlls": ["nvidia/p/bin/p64.dll"]}]}
                m = cuda_pack.parse_manifest(json.dumps(doc), HOSTS, _allow_http_localhost=True)
                if ok:
                    cuda_pack.install(m, Path(d), self.opener)
                else:
                    with self.assertRaises(cuda_pack.PackError):
                        cuda_pack.install(m, Path(d), self.opener)

    def test_real_opener_refuses_http_and_foreign_hosts(self):
        real = cuda_pack.make_opener()
        for url in ("http://files.pythonhosted.org/a.whl", "https://evil.example/a.whl", self.srv.base + "/a.whl"):
            with self.subTest(url), self.assertRaises(cuda_pack.PackError):
                with real(url):
                    pass


class FrozenDllDirs(unittest.TestCase):
    def test_frozen_uses_the_pack_and_source_installs_do_not(self):
        with tempfile.TemporaryDirectory(prefix="parley_test_cuda_pack_gpu_") as d:
            home = Path(d)
            m = cuda_pack.load_manifest()
            pack = home / "runtime" / "cuda12"
            (pack / "bin").mkdir(parents=True)
            files = {}
            for name in m.files:
                (pack / "bin" / name).write_bytes(b"d" * 3)
                files[name] = 3
            (pack / "stamp.json").write_text(json.dumps({"manifest": m.stamp, "files": files}), "utf-8")
            add = getattr(gpu.add_cuda_dll_dirs, "__wrapped__")
            env = {"PARLEY_HOME": d, "PATH": "base"}
            with mock.patch.dict(os.environ, env), mock.patch.object(sys, "frozen", True, create=True), \
                    mock.patch("os.add_dll_directory", create=True) as adder:
                add()
                adder.assert_called_once_with(str(pack / "bin")) if sys.platform == "win32" else None
                self.assertTrue(os.environ["PATH"].startswith(str(pack / "bin")))
            # no pack yet: nothing is added
            shutil.rmtree(pack)
            with mock.patch.dict(os.environ, env), mock.patch.object(sys, "frozen", True, create=True), \
                    mock.patch("os.add_dll_directory", create=True) as adder:
                add()
                adder.assert_not_called()
                self.assertEqual(os.environ["PATH"], "base")

    def test_source_install_ignores_the_pack(self):
        with tempfile.TemporaryDirectory(prefix="parley_test_cuda_pack_src_") as d, \
                mock.patch.dict(os.environ, {"PARLEY_HOME": d}), \
                mock.patch.object(cuda_pack, "usable_dir") as usable:
            self.assertFalse(getattr(sys, "frozen", False))
            getattr(gpu.add_cuda_dll_dirs, "__wrapped__")()
            usable.assert_not_called()


if __name__ == "__main__":
    unittest.main()
