"""llama.cpp download integrity: pinned sha256/size table, verified fetch, safe archive extraction.

Run: uv run python -m unittest discover -s tests -v
No network: downloads are served by a fake stream. Scratch dirs are temp dirs named parley_test_pins_*.
"""
import hashlib
import io
import itertools
import tarfile
import tempfile
import unittest
import zipfile
from contextlib import contextmanager
from pathlib import Path
from unittest import mock

from parley import download, gpu, llama_server
from parley.download import ArchiveError, safe_extract_tar, safe_extract_zip
from parley.llama_server import ASSET_PINS, LLAMA_CPP_TAG, RuntimeDownloadError

SHA = r"[0-9a-f]{64}"


def nv(driver: int):
    return gpu.NvidiaGpu("NVIDIA GeForce RTX 5090", 32607, driver)


def reachable_assets() -> set[str]:
    """Every asset `_build_assets` can return, over platform, arch, runtime and NVIDIA driver combinations."""
    found = set()
    for plat, machine, runtime, driver in itertools.product(
            ("win32", "darwin", "linux"), ("AMD64", "x86_64", "arm64", "aarch64"),
            ("auto", "cuda", "vulkan", "cpu"), (None, 535, 580)):
        with mock.patch.object(llama_server.sys, "platform", plat), \
                mock.patch.object(llama_server.platform, "machine", return_value=machine), \
                mock.patch.object(gpu, "nvidia", return_value=nv(driver) if driver else None):
            found.update(llama_server._build_assets(runtime)[1])
    return found


class PinTable(unittest.TestCase):
    def test_every_reachable_asset_is_pinned_and_nothing_else(self):
        assets = reachable_assets()
        self.assertGreaterEqual(len(assets), 14)
        self.assertEqual(assets - set(ASSET_PINS), set())
        self.assertEqual(set(ASSET_PINS) - assets, set())  # no stale entries

    def test_entries_are_well_formed_and_match_the_tag(self):
        import re
        for name, (sha, size) in ASSET_PINS.items():
            self.assertRegex(sha, SHA, name)
            self.assertGreater(size, 1_000_000, name)
            self.assertLess(size, 2 * 1024**3, name)
            if name.startswith("llama-"):
                self.assertIn(f"-{LLAMA_CPP_TAG}-", name)  # a tag bump without new pins fails here
            self.assertTrue(re.fullmatch(r"[A-Za-z0-9._-]+\.(zip|tar\.gz)", name), name)
        self.assertEqual(len({s for s, _ in ASSET_PINS.values()}), len(ASSET_PINS))

    def test_known_windows_digests(self):  # also verified by a real download when this table was written
        self.assertEqual(ASSET_PINS[f"llama-{LLAMA_CPP_TAG}-bin-win-vulkan-x64.zip"],
                         ("cb34fdf7c628b45adc50954097fba0263a92a9a5413faba1074e790ed8781e14", 32260463))
        self.assertEqual(ASSET_PINS[f"llama-{LLAMA_CPP_TAG}-bin-win-cuda-13.4-x64.zip"],
                         ("39839af853b951bd1454acc458ef7b03d4c8acd6c9da61fa1c29083f011054ea", 154749299))


class Resp:
    def __init__(self, data: bytes):
        self.data = data

    def iter_bytes(self, n=None):
        for i in range(0, len(self.data), 1000):
            yield self.data[i:i + 1000]


def opener(data: bytes, urls: list | None = None):
    @contextmanager
    def open_stream(url):
        if urls is not None:
            urls.append(url)
        yield Resp(data)
    return open_stream


def zip_bytes(members: dict[str, bytes], *, symlink: str | None = None) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for name, data in members.items():
            z.writestr(name, data)
        if symlink:
            zi = zipfile.ZipInfo(symlink)
            zi.create_system = 3
            zi.external_attr = 0o120777 << 16
            z.writestr(zi, "target")
    return buf.getvalue()


def tar_bytes(files: dict[str, bytes], extra: list[tarfile.TarInfo] | None = None) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as t:
        for name, data in files.items():
            ti = tarfile.TarInfo(name)
            ti.size = len(data)
            t.addfile(ti, io.BytesIO(data))
        for ti in extra or []:
            t.addfile(ti)
    return buf.getvalue()


class Scratch(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="parley_test_pins_")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)


class VerifiedFetch(Scratch):
    ASSET = "fake-asset.zip"

    def pin(self, data: bytes, size: int | None = None, sha: str | None = None):
        entry = (sha or hashlib.sha256(data).hexdigest(), len(data) if size is None else size)
        return mock.patch.dict(ASSET_PINS, {self.ASSET: entry})

    def test_good_download_is_kept_and_uses_the_release_url(self):
        data = zip_bytes({"llama-server.exe": b"x"})
        urls = []
        with self.pin(data):
            path = llama_server._fetch_asset(self.ASSET, self.root, opener(data, urls))
        self.assertEqual(path.read_bytes(), data)
        self.assertEqual(urls, [llama_server.RELEASE_URL + self.ASSET])

    def test_tampered_download_is_deleted_and_rejected(self):
        good = zip_bytes({"llama-server.exe": b"x"})
        evil = zip_bytes({"llama-server.exe": b"y"})  # different bytes, same announced sha256 of `good`
        with self.pin(good), self.assertRaises(RuntimeDownloadError) as cm:
            llama_server._fetch_asset(self.ASSET, self.root, opener(evil))
        self.assertIn("safety check", str(cm.exception))
        self.assertEqual(list((self.root / "downloads").iterdir()), [])

    def test_wrong_size_is_rejected(self):
        data = zip_bytes({"a": b"x"})
        for size in (len(data) - 1, len(data) + 1):
            with self.pin(data, size=size), self.assertRaises(RuntimeDownloadError):
                llama_server._fetch_asset(self.ASSET, self.root, opener(data))
            self.assertEqual(list((self.root / "downloads").iterdir()), [])

    def test_unknown_asset_is_refused_without_any_request(self):
        urls = []
        with self.assertRaises(RuntimeDownloadError) as cm:
            llama_server._fetch_asset("llama-evil.zip", self.root, opener(b"x", urls))
        self.assertEqual(urls, [])
        self.assertLess(len(str(cm.exception)), 140)

    def test_a_leftover_file_is_not_trusted(self):
        data = zip_bytes({"a": b"x"})
        (self.root / "downloads").mkdir()
        (self.root / "downloads" / self.ASSET).write_bytes(b"planted")
        with self.pin(data):
            path = llama_server._fetch_asset(self.ASSET, self.root, opener(data))
        self.assertEqual(path.read_bytes(), data)

    def test_release_hosts_only(self):
        op = llama_server.cuda_pack.make_opener(llama_server.RELEASE_HOSTS)
        for url in ("https://evil.example/x.zip", "http://github.com/x.zip", "https://github.com.evil.example/x.zip"):
            with self.assertRaises(llama_server.cuda_pack.PackError), op(url):
                pass


class EnsureRuntime(Scratch):
    def run_ensure(self, data: bytes, name: str = "fake-asset.zip", exe: str = "llama-server.exe"):
        pins = {name: (hashlib.sha256(data).hexdigest(), len(data))}
        with mock.patch.dict(ASSET_PINS, pins, clear=True), \
                mock.patch.object(llama_server, "_build_assets", return_value=("vulkan", [name])), \
                mock.patch.object(llama_server.sys, "platform", "win32" if exe.endswith(".exe") else "linux"):
            return llama_server.ensure_runtime(self.root, "vulkan", opener(data))

    def test_zip_end_to_end(self):
        exe = self.run_ensure(zip_bytes({"llama-server.exe": b"MZ", "ggml.dll": b"d"}))
        self.assertEqual(exe.read_bytes(), b"MZ")
        self.assertTrue((exe.parent / ".complete").exists())
        self.assertEqual(list((self.root / "downloads").iterdir()), [])  # archive removed

    def test_tar_end_to_end(self):
        exe = self.run_ensure(tar_bytes({"llama-b1/llama-server": b"ELF"}), "fake.tar.gz", "llama-server")
        self.assertEqual(exe.read_bytes(), b"ELF")

    def test_corrupt_archives_give_a_short_message_and_leave_nothing(self):
        for name, data in (("fake-asset.zip", b"PK\x03\x04 this is not really a zip" * 4),
                           ("fake.tar.gz", b"\x1f\x8b not a gzip stream at all" * 4)):
            with self.subTest(name=name), self.assertLogs(llama_server.log, "WARNING"), \
                    self.assertRaises(RuntimeDownloadError) as cm:
                self.run_ensure(data, name)
            self.assertIn("could not be unpacked", str(cm.exception))
            self.assertNotIn("Traceback", str(cm.exception))
            runtime = self.root / "runtime"
            self.assertFalse(any(runtime.rglob(".complete")))
            self.assertEqual(list(runtime.rglob("*")), [])
            self.assertEqual(list((self.root / "downloads").iterdir()), [])

    def test_oserror_during_extraction_is_wrapped(self):
        data = zip_bytes({"llama-server.exe": b"MZ"})
        with mock.patch.object(llama_server, "safe_extract_zip", side_effect=OSError(28, "No space left")), \
                self.assertLogs(llama_server.log, "WARNING"), self.assertRaises(RuntimeDownloadError):
            self.run_ensure(data)
        self.assertEqual(list((self.root / "runtime").rglob("*")), [])

    def test_hostile_zip_is_rejected_and_nothing_is_left(self):
        with self.assertRaises(RuntimeDownloadError) as cm:
            self.run_ensure(zip_bytes({"llama-server.exe": b"MZ", "../evil.dll": b"x"}))
        self.assertIn("unsafe", str(cm.exception))
        runtime = self.root / "runtime"
        self.assertFalse(any(runtime.rglob("*.exe")) or any(runtime.rglob(".complete")))
        self.assertFalse((self.root / "evil.dll").exists())
        self.assertEqual(list((self.root / "downloads").iterdir()), [])


class ZipExtraction(Scratch):
    def extract(self, data: bytes) -> Path:
        z = self.root / "a.zip"
        z.write_bytes(data)
        out = self.root / "out"
        out.mkdir(exist_ok=True)
        safe_extract_zip(z, out)
        return out

    def test_normal_archive(self):
        out = self.extract(zip_bytes({"llama-server.exe": b"a", "ggml-cuda.dll": b"b", "sub/dir/x_y-1.2+z.txt": b"c", "d/": b""}))
        self.assertEqual((out / "llama-server.exe").read_bytes(), b"a")
        self.assertEqual((out / "sub" / "dir" / "x_y-1.2+z.txt").read_bytes(), b"c")
        self.assertTrue((out / "d").is_dir())

    def test_hostile_names_rejected_and_nothing_written(self):
        bad = ["../evil.dll", "a/../../evil.dll", "/abs/evil.dll", "C:/evil.dll", "C:evil.dll", "\\\\srv\\share\\x",
               "a/./b.dll", "a//b.dll", "con.dll", "NUL", "a/aux.txt", "x.", "a b.dll", "caf\u00e9.dll", "evil:stream",
               "p" * 200, "a/b/c/d/e/f/g/h.dll", ""]
        for name in bad:
            with self.subTest(name=name), tempfile.TemporaryDirectory(prefix="parley_test_pins_") as d:
                if not name:
                    with self.assertRaises(ArchiveError):
                        download._check_name(name)
                    continue
                zi = zipfile.ZipInfo("placeholder")
                zi.filename = name  # set afterwards: the constructor would normalise "\\" and cut at NUL
                buf = io.BytesIO()
                with zipfile.ZipFile(buf, "w") as zf:
                    zf.writestr("good.dll", b"ok")
                    zf.writestr(zi, b"x")
                z, out = Path(d) / "h.zip", Path(d) / "o"
                z.write_bytes(buf.getvalue())
                out.mkdir()
                with self.assertRaises(ArchiveError):
                    safe_extract_zip(z, out)
                self.assertEqual(list(out.iterdir()), [], "validation happens before any write")

    def test_names_zipfile_would_normalise_are_rejected_by_the_rule_itself(self):
        # zipfile turns "\\" into "/" on Windows and cuts at NUL when it reads, but not on other platforms
        for name in ("a\\b.dll", "a\x00b", "..\\evil.dll"):
            with self.assertRaises(ArchiveError):
                download._check_name(name)

    def test_symlink_entry_rejected(self):
        with self.assertRaises(ArchiveError):
            self.extract(zip_bytes({"a.dll": b"x"}, symlink="link"))

    def test_duplicate_and_case_duplicate_rejected(self):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as z:
            z.writestr("a.dll", b"1")
            z.writestr("A.DLL", b"2")
        with self.assertRaises(ArchiveError):
            self.extract(buf.getvalue())

    def test_oversized_member_rejected_by_declared_size(self):
        with mock.patch.object(download, "MAX_MEMBER_BYTES", 10), self.assertRaises(ArchiveError):
            self.extract(zip_bytes({"big.bin": b"x" * 11}))

    def test_total_and_count_caps(self):
        with mock.patch.object(download, "MAX_TOTAL_BYTES", 15), self.assertRaises(ArchiveError):
            self.extract(zip_bytes({"a": b"x" * 10, "b": b"y" * 10}))
        with mock.patch.object(download, "MAX_MEMBERS", 1), self.assertRaises(ArchiveError):
            self.extract(zip_bytes({"a": b"x", "b": b"y"}))


class TarExtraction(Scratch):
    def extract(self, data: bytes) -> Path:
        t = self.root / "a.tar.gz"
        t.write_bytes(data)
        out = self.root / "out"
        safe_extract_tar(t, out)
        return out

    def test_normal_archive_with_relative_symlink(self):
        link = tarfile.TarInfo("llama/libx.0.dylib")
        link.type, link.linkname = tarfile.SYMTYPE, "libx.0.27.dylib"
        out = self.extract(tar_bytes({"llama/llama-server": b"a", "llama/libx.0.27.dylib": b"b"}, [link]))
        self.assertEqual((out / "llama" / "llama-server").read_bytes(), b"a")

    def test_hostile_members_rejected(self):
        def ti(name, typ=tarfile.REGTYPE, link=""):
            t = tarfile.TarInfo(name)
            t.type, t.linkname = typ, link
            return t
        cases = {
            "dotdot": ti("../evil"), "abs": ti("/etc/evil"), "drive": ti("C:/evil"), "backslash": ti("a\\b"),
            "symlink out": ti("l", tarfile.SYMTYPE, "../../etc/passwd"), "symlink abs": ti("l", tarfile.SYMTYPE, "/etc/passwd"),
            "symlink sub": ti("l", tarfile.SYMTYPE, "a/b"), "hardlink": ti("h", tarfile.LNKTYPE, "good"),
            "device": ti("dev", tarfile.CHRTYPE), "fifo": ti("f", tarfile.FIFOTYPE),
        }
        for label, member in cases.items():
            with self.subTest(label), tempfile.TemporaryDirectory(prefix="parley_test_pins_") as d:
                t = Path(d) / "h.tar.gz"
                t.write_bytes(tar_bytes({"good": b"ok"}, [member]))
                out = Path(d) / "out"
                with self.assertRaises(ArchiveError):
                    safe_extract_tar(t, out)
                self.assertEqual(list(out.iterdir()), [])

    def test_oversized_member_rejected(self):
        with mock.patch.object(download, "MAX_MEMBER_BYTES", 10), self.assertRaises(ArchiveError):
            self.extract(tar_bytes({"big": b"x" * 11}))


if __name__ == "__main__":
    unittest.main()
