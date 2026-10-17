"""The CUDA pack: CUDA 12 libraries a frozen (packaged) Parley downloads on first run on an NVIDIA machine.

A source checkout gets these libraries from the `nvidia-*-cu12` pip wheels; the packaged app leaves them out (about 2 GB)
and fetches the same pinned wheels instead, into `<Parley home>/runtime/cuda12/bin/`. The home is outside the versioned
app tree, so the pack survives app updates and is shared by every installed version.

`cuda_pack.json` (generated from uv.lock by scripts/cuda_pack.py, bundled in the build) lists each wheel's URL, sha256,
size and the DLLs to take from it. Security (this downloads and loads native code): https only and only the hosts in
ALLOWED_HOSTS, redirects only between those hosts, every wheel size- and sha256-checked before it is opened (a mismatch
deletes it), only the listed DLL names are extracted (strict names, exclusive create, size caps) and everything else in a
wheel is ignored, the pack is built in a partial directory and renamed into place with a stamp file written last, so a
crash or a failed download never leaves a half pack. Any failure is a one-line log and Parley stays on the CPU; the next
launch tries again. The pack is used from the next start after it completes (the DLL search path is set once at startup).
"""
import hashlib
import json
import logging
import os
import re
import shutil
import sys
import threading
import zipfile
from collections import Counter
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urljoin, urlsplit

import httpx

from . import __version__, update
from .download import download_verified

log = logging.getLogger(__name__)

PACK_NAME = "cuda12"
BIN = "bin"
STAMP_NAME = "stamp.json"
MANIFEST_PATH = Path(__file__).with_name("cuda_pack.json")
ALLOWED_HOSTS = frozenset({"files.pythonhosted.org"})  # the hosts of the wheel URLs in uv.lock
PLATFORM = "win_amd64"
MAX_WHEELS = 16
MAX_WHEEL_BYTES = 2 * 1024**3
MAX_DLL_BYTES = 1024**3  # largest real DLL is cublasLt at about 0.67 GB
MAX_PACK_BYTES = 6 * 1024**3
MAX_REDIRECTS = 3
_MEMBER = re.compile(r"nvidia/[a-z0-9_]{1,40}/bin/[A-Za-z0-9._-]{1,100}\.dll", re.ASCII)
_HEX64 = re.compile(r"[0-9a-f]{64}")


class PackError(Exception):
    """The manifest or a wheel is not acceptable, or another process holds the pack."""


@dataclass(frozen=True)
class Wheel:
    package: str
    url: str
    sha256: str
    size: int
    dlls: tuple[str, ...]  # member paths inside the wheel


@dataclass(frozen=True)
class Manifest:
    wheels: tuple[Wheel, ...]
    stamp: str  # sha256 of the canonical manifest; a different stamp means a different pack

    @property
    def total_download(self) -> int:
        return sum(w.size for w in self.wheels)

    @property
    def files(self) -> frozenset[str]:
        return frozenset(d.rsplit("/", 1)[1] for w in self.wheels for d in w.dlls)


# ---------- manifest ----------

def _url_ok(url: object, hosts: frozenset[str] | set[str], http_local: bool) -> str:
    if not isinstance(url, str) or len(url) > 1024:
        raise PackError("bad URL")
    try:
        u = urlsplit(url)
        host = u.hostname
        u.port  # noqa: B018  (raises ValueError on a bad port)
    except ValueError as e:
        raise PackError("bad URL") from e
    if not host or u.username or u.password or host not in hosts:
        raise PackError(f"download host not allowed: {host}")
    if u.scheme != "https" and not (http_local and u.scheme == "http" and host in ("127.0.0.1", "localhost")):
        raise PackError("downloads need https")
    return url


def parse_manifest(raw: bytes | str, allowed_hosts: frozenset[str] | set[str] = ALLOWED_HOSTS,
                   *, _allow_http_localhost: bool = False) -> Manifest:
    """Validate `cuda_pack.json`. `_allow_http_localhost` is a test hook only."""
    try:
        doc = json.loads(raw)
    except ValueError as e:
        raise PackError("manifest is not JSON") from e
    if not isinstance(doc, dict) or doc.get("schema") != 1 or doc.get("platform") != PLATFORM:
        raise PackError("unsupported manifest")
    items = doc.get("wheels")
    if not isinstance(items, list) or not 0 < len(items) <= MAX_WHEELS:
        raise PackError("bad wheel list")
    wheels, urls, names = [], set(), set()
    for it in items:
        if not isinstance(it, dict):
            raise PackError("bad wheel entry")
        url = _url_ok(it.get("url"), allowed_hosts, _allow_http_localhost)
        sha, size, dlls = it.get("sha256"), it.get("size"), it.get("dlls")
        if not isinstance(sha, str) or not _HEX64.fullmatch(sha):
            raise PackError("bad sha256")
        if not isinstance(size, int) or isinstance(size, bool) or not 0 < size <= MAX_WHEEL_BYTES:
            raise PackError("bad wheel size")
        if not isinstance(dlls, list) or not 0 < len(dlls) <= 32:
            raise PackError("bad DLL list")
        for d in dlls:
            if not isinstance(d, str) or not _MEMBER.fullmatch(d):
                raise PackError("bad DLL path")
            try:
                update.validate_rel_path(d)
            except update.UpdateError as e:
                raise PackError("bad DLL path") from e
            key = d.rsplit("/", 1)[1].casefold()
            if key in names:
                raise PackError("duplicate DLL name")
            names.add(key)
        if url in urls:
            raise PackError("duplicate wheel")
        urls.add(url)
        package = it.get("package")
        wheels.append(Wheel(package if isinstance(package, str) else "?", url, sha, size, tuple(dlls)))
    stamp = hashlib.sha256(json.dumps(doc, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return Manifest(tuple(wheels), stamp)


def load_manifest(path: Path = MANIFEST_PATH) -> Manifest | None:
    """The bundled manifest, or None (missing or invalid: logged once, then Parley just stays on the CPU)."""
    try:
        return parse_manifest(path.read_bytes())
    except (OSError, PackError) as e:
        log.warning("CUDA pack manifest unusable (%s)", e)
        return None


# ---------- pack state ----------

def pack_dir(home: Path) -> Path:
    return home / "runtime" / PACK_NAME


def usable_dir(home: Path, manifest: Manifest | None = None) -> Path | None:
    """`<pack>/bin` if a complete pack for this manifest is installed, else None. Checks the stamp and the size of
    every file (cheap; nothing is re-hashed at startup)."""
    m = manifest or load_manifest()
    if m is None:
        return None
    root = pack_dir(home)
    try:
        stamp = json.loads((root / STAMP_NAME).read_text("utf-8"))
        files = stamp["files"]
        if stamp["manifest"] != m.stamp or not isinstance(files, dict) or set(files) != set(m.files):
            return None
        for name, size in files.items():
            p = root / BIN / name
            if os.path.islink(p) or not p.is_file() or p.stat().st_size != size:
                return None
    except (OSError, ValueError, KeyError, TypeError):
        return None
    return root / BIN


# ---------- download ----------

def make_opener(allowed_hosts: frozenset[str] | set[str] = ALLOWED_HOSTS, *, _allow_http_localhost: bool = False):
    """open_stream(url) for `download_verified`: certificate checks, no automatic redirects, redirects only to allowed
    https hosts, overall deadline and minimum rate (update._Paced)."""
    @contextmanager
    def open_stream(url: str):
        headers = {"User-Agent": f"parley/{__version__} cuda-pack"}
        with httpx.Client(timeout=update.TIMEOUT, follow_redirects=False, headers=headers) as client:
            for _ in range(MAX_REDIRECTS + 1):
                _url_ok(url, allowed_hosts, _allow_http_localhost)
                r = client.send(client.build_request("GET", url), stream=True)
                if r.is_redirect:
                    loc = r.headers.get("location")
                    r.close()
                    if not loc:
                        raise PackError("bad redirect")
                    url = urljoin(url, loc)
                    continue
                try:
                    r.raise_for_status()
                    yield update._Paced(r)
                finally:
                    r.close()
                return
            raise PackError("too many redirects")
    return open_stream


@contextmanager
def _lock(path: Path):
    """Exclusive, non-blocking OS lock (released automatically if the process dies), so two Parley instances
    never build the pack in the same directory. Raises PackError if it is held."""
    path.parent.mkdir(parents=True, exist_ok=True)
    f = open(path, "a+b")
    try:
        try:
            if sys.platform == "win32":
                import msvcrt
                msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as e:
            raise PackError("another Parley is already downloading GPU support") from e
        yield
    finally:
        f.close()


def extract_dlls(wheel: Path, w: Wheel, bin_dir: Path) -> dict[str, int]:
    """Copy exactly the DLLs listed for `w` out of the wheel into `bin_dir` (flat). Everything else in the zip is
    ignored. Exclusive create (never overwrites), declared and actual sizes capped. Returns {name: size}."""
    done: dict[str, int] = {}
    with zipfile.ZipFile(wheel) as zf:
        counts = Counter(i.filename for i in zf.infolist())
        for member in w.dlls:
            if counts[member] != 1:
                raise PackError(f"{w.package}: {member} is missing or listed twice in the wheel")
            info = zf.getinfo(member)
            if info.is_dir() or info.file_size > MAX_DLL_BYTES:
                raise PackError(f"{w.package}: {member} is not an acceptable file")
            name = member.rsplit("/", 1)[1]
            dest = bin_dir / name
            written = 0
            with zf.open(info) as src, open(dest, "xb") as out:
                while chunk := src.read(1 << 20):
                    written += len(chunk)
                    if written > min(info.file_size, MAX_DLL_BYTES):
                        raise PackError(f"{w.package}: {member} is larger than it declares")
                    out.write(chunk)
            if written != info.file_size:
                raise PackError(f"{w.package}: {member} is truncated")
            done[name] = written
    return done


def install(m: Manifest, home: Path, open_stream: Callable, *, on_start: Callable[[], None] | None = None) -> Path:
    """Download and extract the whole pack into `<home>/runtime/cuda12`, atomically. Returns `<pack>/bin`.
    Raises on any failure after removing the partial directory; an existing pack is replaced only on success."""
    runtime = home / "runtime"
    final, partial, old = (runtime / (PACK_NAME + s) for s in ("", ".partial", ".old"))
    runtime.mkdir(parents=True, exist_ok=True)
    with _lock(runtime / (PACK_NAME + ".lock")):
        shutil.rmtree(partial, ignore_errors=True)  # left by a crashed earlier run
        need = m.total_download * 2 + (256 << 20)  # wheels are deleted one by one, the extracted DLLs stay
        free = shutil.disk_usage(runtime).free
        if free < need:
            raise PackError(f"not enough disk space ({free // 2**20} MB free, {need // 2**20} MB needed)")
        if on_start:
            on_start()
        bin_dir, dl = partial / BIN, partial / ".dl"
        try:
            bin_dir.mkdir(parents=True)
            dl.mkdir()
            files: dict[str, int] = {}
            for i, w in enumerate(m.wheels, 1):
                log.info("Downloading GPU support %d of %d: %s (%d MB)", i, len(m.wheels), w.package, w.size // 2**20)
                path = download_verified(open_stream, w.url, dl / f"{i}.whl", w.sha256, w.size)
                files.update(extract_dlls(path, w, bin_dir))
                path.unlink()
                if sum(files.values()) > MAX_PACK_BYTES:
                    raise PackError("pack too large")
            shutil.rmtree(dl)
            tmp = partial / (STAMP_NAME + ".tmp")
            tmp.write_text(json.dumps({"manifest": m.stamp, "files": files}), "utf-8")
            tmp.replace(partial / STAMP_NAME)  # last: no stamp, no usable pack
            shutil.rmtree(old, ignore_errors=True)
            if final.exists():
                final.rename(old)
            try:
                partial.rename(final)
            except OSError:
                if old.exists() and not final.exists():
                    old.rename(final)
                raise
            shutil.rmtree(old, ignore_errors=True)
        except BaseException:
            shutil.rmtree(partial, ignore_errors=True)
            raise
    return final / BIN


def needed(cfg, plan) -> bool:
    """Would a stage run on CUDA? Forced `cpu` (stt.device / tts.device) or `[gpu] download_cuda = false` say no, and
    so does a plan without NVIDIA. (llama.cpp ships its own CUDA runtime; it does not use this pack.)"""
    if not cfg.gpu.download_cuda:
        return False
    return (cfg.stt.device != "cpu" and plan.stt == "cuda") or (cfg.tts.device != "cpu" and plan.tts == "cuda")


def is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False)) and sys.platform == "win32"


def _size_words(n: int) -> str:
    return f"{n / 1e9:.1f} gigabytes" if n >= 10**9 else f"{max(1, round(n / 1e8))} hundred megabytes"


def maybe_download(cfg, plan, home: Path, say: Callable[[str], None] | None = None, *,
                   frozen: bool | None = None, manifest: Manifest | None = None, open_stream: Callable | None = None) -> bool:
    """Fetch the pack if this is a packaged copy on an NVIDIA machine that would use it and doesn't have it.
    Never raises; True only when a new pack was installed. Blocking: run it in a thread (`start_background`)."""
    try:
        if not (is_frozen() if frozen is None else frozen) or not needed(cfg, plan):
            return False
        m = manifest or load_manifest()
        if m is None or usable_dir(home, m):
            return False

        def announce() -> None:
            log.info("Downloading GPU support (%d MB); using the CPU until it is done", m.total_download // 2**20)
            if say:
                say(f"Downloading GPU support, about {_size_words(m.total_download)}. "
                    "Parley will use the CPU until it is done.")

        install(m, home, open_stream or make_opener(), on_start=announce)
        log.info("GPU support installed in %s; it is used from the next start", pack_dir(home))
        if say:
            say("GPU support is ready. Parley will use it the next time it starts.")
        return True
    except PackError as e:
        log.warning("GPU support not installed: %s. Parley stays on the CPU and will try again next start", e)
    except Exception as e:  # offline, hash mismatch, disk full, bad zip ...: never stop Parley
        log.warning("GPU support download failed (%s: %s). Parley stays on the CPU and will try again next start",
                    type(e).__name__, e)
        log.debug("CUDA pack failure detail", exc_info=True)
    return False


def start_background(cfg, plan, home: Path, say: Callable[[str], None] | None = None) -> threading.Thread | None:
    """Run `maybe_download` in a daemon thread when it could do anything; None otherwise."""
    if not (is_frozen() and needed(cfg, plan)):
        return None
    t = threading.Thread(target=maybe_download, args=(cfg, plan, home, say), daemon=True, name="cuda-pack")
    t.start()
    return t
