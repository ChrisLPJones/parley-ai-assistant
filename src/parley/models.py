"""Model catalog, hardware fit and the download manager behind the Models page (no UI, no network at import).

Security (downloads): a model is fetched only by catalog id, never from a URL or path given by the caller. Every file has a
pinned sha256 and size (`model_pins.PINS`) and a fixed Hugging Face commit; https only, exact host set (also after
redirects), `download_verified` (size cap, sha256, `.part` then atomic replace). A model without a pin is PENDING_PIN and
is refused. Delete is confined to `<home>/models`, refuses the model in use and never follows links.
"""
import hashlib
import logging
import os
import re
import shutil
import sys
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

import httpx

from . import cuda_pack, download, gpu, llama_server, update
from .download import DownloadCancelled, DownloadError, DownloadTruncated, download_verified
from .model_pins import PINS

log = logging.getLogger(__name__)

PENDING_PIN = "PENDING_PIN"  # state of a catalog entry whose pins are not in model_pins.PINS yet
MAX_MODEL_BYTES = 12 * 1024**3
MIN_MODEL_BYTES = 1024**2
FREE_MARGIN = 2 * 1024**3  # kept free on the drive on top of the download (a guess)
# Hugging Face and its CDN. Exact names, no wildcard. `scripts/pin_models.py` prints the hosts it really observes in the
# redirects so this list can be checked against reality (the CDN names below are not yet verified).
# `us.aws.cdn.hf.co` was observed on 2026-10-19 by scripts/pin_models.py as the redirect target of every LFS file; the
# other CDN names are guesses kept until the script output shows they are never used.
HF_HOSTS = frozenset({"huggingface.co", "us.aws.cdn.hf.co", "cdn-lfs.huggingface.co", "cdn-lfs.hf.co",
                      "cdn-lfs-us-1.hf.co", "cas-bridge.xethub.hf.co"})
KOKORO_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/"
KOKORO_HOSTS = llama_server.RELEASE_HOSTS
NAME_RE = re.compile(r"[A-Za-z0-9._-]{1,128}", re.ASCII)
HEX64 = re.compile(r"[0-9a-f]{64}")
KIND_DIR = {"llm": "llm", "stt": "whisper", "tts": "kokoro"}
# Presets that older versions of Parley already downloaded without a pin; until their pins are generated they keep
# working through the old (unverified) path. Any other preset without a pin is refused.
LEGACY_UNPINNED = frozenset({"Qwen3.5-4B-Q4_K_M.gguf", "Qwen3.5-2B-Q4_K_M.gguf"})
_DEVICES = llama_server._WIN_DEVICES


class ModelError(Exception):
    """A refusal or failure with a short message that can be shown or spoken as it is."""


@dataclass(frozen=True)
class File:
    name: str
    url: str
    sha256: str
    size: int


@dataclass(frozen=True)
class Model:
    id: str
    kind: str  # llm | stt | tts
    label: str
    note: str
    hosts: frozenset[str]
    approx_bytes: int  # shown until the pins give the real sizes
    min_mem_gb: float
    key: str  # preset name / Whisper model name / Kokoro variant
    probe: tuple[str, ...]  # file names to look for while the model has no pin
    files: tuple[File, ...] = ()  # empty = PENDING_PIN
    subdir: str = ""  # Whisper: models/whisper/<subdir>/

    @property
    def pinned(self) -> bool:
        return bool(self.files)

    @property
    def total_bytes(self) -> int:
        return sum(f.size for f in self.files) if self.files else self.approx_bytes

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(f.name for f in self.files) or self.probe


# ---------- catalog ----------

def _llm(key: str, label: str, note: str, approx_mb: int) -> dict:
    repo, file, gb = llama_server.PRESETS[key]
    return dict(id=f"llm-{key}", kind="llm", label=label, note=note, hosts=HF_HOSTS, approx_bytes=approx_mb * 1024**2,
                min_mem_gb=float(gb), key=key, probe=(file,), repo=repo)


def _stt(name: str, repo: str, approx_mb: int, gb: float, note: str) -> dict:
    return dict(id=f"stt-{name}", kind="stt", label=f"Whisper {name}", note=note, hosts=HF_HOSTS,
                approx_bytes=approx_mb * 1024**2, min_mem_gb=gb, key=name, probe=("model.bin",), subdir=name, repo=repo)


def _tts(key: str, label: str, note: str, file: str, approx_mb: int) -> dict:
    return dict(id="tts-kokoro" if key == "default" else f"tts-kokoro-{key}", kind="tts", label=label, note=note,
                hosts=KOKORO_HOSTS, approx_bytes=approx_mb * 1024**2, min_mem_gb=1.0, key=key,
                probe=(file, "voices-v1.0.bin"), repo="")


# Approximate sizes (MB) are until the pins are generated.
SPECS: tuple[dict, ...] = (
    _llm("tiny", "Qwen3.5 2B (tiny)", "Fastest; fine on a laptop CPU.", 1300),
    _llm("small", "Qwen3.5 4B (small)", "A good balance; best with a graphics card.", 2700),
    _llm("medium", "Qwen3.5 9B (medium)", "Smarter answers; wants a strong graphics card or plenty of memory.", 5700),
    _stt("tiny.en", "Systran/faster-whisper-tiny.en", 75, 1.0, "Smallest; English only."),
    _stt("base.en", "Systran/faster-whisper-base.en", 145, 1.0, "Small and quick; English only."),
    _stt("small.en", "Systran/faster-whisper-small.en", 484, 1.5, "Good on a CPU; English only."),
    _stt("small", "Systran/faster-whisper-small", 484, 1.5, "Good on a CPU; many languages."),
    _stt("medium", "Systran/faster-whisper-medium", 1500, 3.0, "More accurate; slow without a graphics card."),
    # The old repo name mobiuslabsgmbh/faster-whisper-large-v3-turbo now 307-redirects to dropbox-dash/... (seen in the HF
    # API; the owner change is not independently confirmed). The content is pinned by commit + sha256, not by the name.
    _stt("large-v3-turbo", "dropbox-dash/faster-whisper-large-v3-turbo", 1600, 3.0,
         "Most accurate; made for a graphics card."),
    _tts("default", "Kokoro (standard)", "The default voice model.", "kokoro-v1.0.onnx", 354),
    _tts("fp16", "Kokoro (half precision)", "Smaller; check that the voice sounds right to you.",
         "kokoro-v1.0.fp16.onnx", 197),
    _tts("int8", "Kokoro (8-bit)", "Smallest; the voice may sound rougher.", "kokoro-v1.0.int8.onnx", 120),
)


def _file_url(spec: dict, revision: str, name: str) -> str:
    if spec["kind"] == "tts":
        return KOKORO_URL + name
    return f"https://huggingface.co/{spec['repo']}/resolve/{revision}/{name}"


def build_catalog(pins: dict[str, dict] | None = None) -> dict[str, Model]:
    """Model objects for SPECS; an entry with a pin gets its files (URL at the pinned commit), the rest stay PENDING_PIN."""
    pins = PINS if pins is None else pins
    out: dict[str, Model] = {}
    for spec in SPECS:
        fields = {k: v for k, v in spec.items() if k != "repo"}
        pin = pins.get(spec["id"])
        files: tuple[File, ...] = ()
        if pin:
            files = tuple(File(n, _file_url(spec, pin["revision"], n), sha, size) for n, sha, size in pin["files"])
        out[spec["id"]] = Model(files=files, **fields)
    return out


CATALOG: dict[str, Model] = build_catalog()


def llm_for_file(file: str, catalog: dict[str, Model] | None = None) -> Model | None:
    for m in (CATALOG if catalog is None else catalog).values():
        if m.kind == "llm" and file in m.names:
            return m
    return None


def pending(catalog: dict[str, Model] | None = None) -> list[str]:
    """Ids of the entries that still need pins (PENDING_PIN)."""
    return [m.id for m in (CATALOG if catalog is None else catalog).values() if not m.pinned]


def validate_model(m: Model, *, allow_http_localhost: bool = False) -> None:
    """Raise ModelError unless the entry is safe to download: plain ids and file names, https, host in the entry's exact
    host set (no userinfo or port), sane sizes, well-formed hashes. The tests run it over the whole catalog and the
    manager before every download."""
    if not NAME_RE.fullmatch(m.id) or m.kind not in KIND_DIR:
        raise ModelError("bad catalog entry")
    if m.subdir and (not NAME_RE.fullmatch(m.subdir) or m.subdir in (".", "..")
                     or m.subdir.split(".")[0].lower() in _DEVICES):
        raise ModelError("bad catalog entry")
    if not MIN_MODEL_BYTES <= m.total_bytes <= MAX_MODEL_BYTES:
        raise ModelError("bad catalog size")
    seen = set()
    for f in m.files:
        if (not NAME_RE.fullmatch(f.name) or f.name.startswith(".") or f.name.split(".")[0].lower() in _DEVICES
                or f.name in seen):
            raise ModelError("bad catalog file name")
        seen.add(f.name)
        if not HEX64.fullmatch(f.sha256) or not 0 < f.size <= MAX_MODEL_BYTES:
            raise ModelError("bad catalog pin")
        try:
            u = urlsplit(f.url)
            host, port = u.hostname, u.port
        except ValueError as e:
            raise ModelError("bad catalog URL") from e
        local = allow_http_localhost and u.scheme == "http" and host in ("127.0.0.1", "localhost")
        if (u.scheme != "https" and not local) or u.username or u.password or host not in m.hosts:
            raise ModelError("bad catalog URL")
        if not local and port not in (None, 443):
            raise ModelError("bad catalog URL")
        if ".." in u.path.split("/") or u.fragment or u.query:
            raise ModelError("bad catalog URL")


# ---------- hardware and fit ----------

@dataclass(frozen=True)
class Hardware:
    vram_mb: int = 0  # video memory of the GPU the plan uses (0 = none usable)
    gpu_name: str = ""
    ram_mb: int = 0  # 0 = unknown
    free_disk_mb: int = 0
    llm_runtime: str = "cpu"  # cuda | vulkan | cpu
    speech_runtime: str = "cpu"  # cuda | cpu
    preset: str = "tiny"  # what the hardware plan picks
    language: str = "en"


def _ram_mb() -> int:
    """Installed RAM in MB, 0 when it cannot be read (no new dependency: ctypes on Windows, sysconf elsewhere)."""
    try:
        if sys.platform == "win32":
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
            st = MEMORYSTATUSEX()
            st.dwLength = ctypes.sizeof(st)
            if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
                return 0
            return int(st.ullTotalPhys // 1024**2)
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") // 1024**2
    except (OSError, ValueError, AttributeError):
        return 0


def free_bytes(path: Path) -> int:
    """Free space on the drive holding `path` (or its nearest existing parent)."""
    p = path
    while not p.exists() and p != p.parent:
        p = p.parent
    return shutil.disk_usage(p).free


def detect_hardware(home: Path, language: str = "en") -> Hardware:
    """Probe this computer (nvidia-smi, DXGI, memory, disk). Can take a moment: call it from a worker thread."""
    plan = gpu.current_plan()
    nv = gpu.nvidia()
    vram, name = 0, ""
    if nv:
        vram, name = nv.vram_mb, nv.name
    elif plan.llm == "vulkan":
        best = max((a for a in gpu.adapters() if a.vendor in ("amd", "intel")), key=lambda a: a.dedicated_mb,
                   default=None)
        if best:
            vram, name = best.dedicated_mb, best.name
    try:
        disk = free_bytes(home) // 1024**2
    except OSError:
        disk = 0
    return Hardware(vram, name, _ram_mb(), disk, plan.llm, plan.stt, plan.preset, language)


def _cpu_fit(hw: Hardware, need_gb: float, words: str, level: str) -> tuple[str, str]:
    if hw.ram_mb <= 0:
        return level, words if level == "good" else "Could not check your memory, may run slowly"
    if hw.ram_mb / 1024 >= 2 * need_gb:
        return level, words
    return "no", "Too big for this computer"


def suitability(m: Model, hw: Hardware) -> tuple[str, str]:
    """(level, words) for one model on this computer; level is good | slow | no. It never hides a model: the page shows
    them all with this verdict, and the user may still pick one at their own risk."""
    need = m.min_mem_gb
    if m.kind == "llm":
        if hw.llm_runtime in ("cuda", "vulkan"):
            avail = (hw.vram_mb - (gpu.SPEECH_VRAM_MB if hw.llm_runtime == "cuda" else 0)) / 1024
            if avail >= need:
                return "good", "Fits your GPU"
        return _cpu_fit(hw, need, "Runs on the CPU, slower", "slow")
    if hw.speech_runtime == "cuda" and hw.vram_mb / 1024 >= need:
        return "good", "Fits your GPU"
    if need <= 1.5:
        return _cpu_fit(hw, need, "Runs on the CPU", "good")
    return _cpu_fit(hw, need, "Runs on the CPU, slower", "slow")


def recommended_ids(hw: Hardware) -> set[str]:
    """The model per kind the page highlights for this computer."""
    stt = "large-v3-turbo" if hw.speech_runtime == "cuda" else ("small.en" if hw.language == "en" else "small")
    return {f"llm-{hw.preset}", f"stt-{stt}", "tts-kokoro"}


def ids_in_use(cfg, home: Path, stt_name: str | None = None, catalog: dict[str, Model] | None = None) -> set[str]:
    """Catalog ids the running (or configured) app uses: the LLM file, the Whisper name and the Kokoro model."""
    cat = CATALOG if catalog is None else catalog
    used = {"tts-kokoro"}
    if stt_name:
        used.add(f"stt-{stt_name}")
    try:
        path, _, _ = llama_server.resolve_model(cfg.llm, home)
        used |= {m.id for m in cat.values() if m.kind == "llm" and path.name in m.names}
    except SystemExit:
        pass
    return used & set(cat)


@dataclass(frozen=True)
class OwnFile:
    """The user's own LLM file (llm.model_path or a custom repo): shown read-only, never deletable."""
    label: str
    path: Path
    bytes: int


def own_llm_file(cfg, home: Path) -> OwnFile | None:
    if not (cfg.llm.model_path or cfg.llm.model_repo):
        return None
    try:
        path, _, _ = llama_server.resolve_model(cfg.llm, home)
        return OwnFile("Your own file", path, path.stat().st_size)
    except (SystemExit, OSError):
        return None


# ---------- paths ----------

def _is_link(p: Path) -> bool:
    """Symlink, or a Windows junction / other reparse point."""
    try:
        if os.path.islink(p):
            return True
        return bool(getattr(os.lstat(p), "st_file_attributes", 0) & 0x400)
    except OSError:
        return False


def model_dir(home: Path, m: Model) -> Path:
    base = home / "models" / KIND_DIR[m.kind]
    return base / m.subdir if m.subdir else base


def _confined(home: Path, p: Path) -> Path:
    """`p` must stay inside <home>/models and no existing component below `models` may be a link."""
    root = home / "models"
    if not p.resolve().is_relative_to(root.resolve()):
        raise ModelError("That file is outside Parley's models folder.")
    cur = root
    for part in p.relative_to(root).parts:
        cur = cur / part
        if _is_link(cur):
            raise ModelError("A models folder entry is a link, so Parley will not touch it.")
    return p


# ---------- manager ----------

@dataclass(frozen=True)
class Status:
    state: str  # not_installed | installed | partial | downloading | legacy
    bytes: int = 0


@dataclass(frozen=True)
class DownloadState:
    model_id: str = ""
    file_i: int = 0
    file_count: int = 0
    done: int = 0
    total: int = 0
    phase: str = "idle"  # idle | downloading | verifying | done | error | cancelled
    message: str = ""


def _size(p: Path) -> int | None:
    try:
        return p.stat().st_size if p.is_file() and not _is_link(p) else None
    except OSError:
        return None


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def _explain(e: BaseException, label: str) -> str:
    """A short message without URLs or response bodies."""
    if isinstance(e, DownloadTruncated):
        return f"The {label} download ended early. Try again to continue where it stopped."
    if isinstance(e, DownloadError):
        return f"The {label} download failed its safety check, so Parley discarded it."
    if isinstance(e, cuda_pack.PackError):
        return f"The {label} download was refused: it did not come from an expected server."
    if isinstance(e, update.UpdateError):
        return f"The {label} download was too slow and was stopped. You can try again."
    if isinstance(e, (httpx.HTTPError, OSError)):
        return f"The {label} download could not finish. Check your connection and the free disk space, then try again."
    return f"The {label} download failed."


class ModelManager:
    """Scan, download (one at a time, in a worker thread) and delete catalog models. The UI only calls these methods and
    reads `state`; it never passes a URL or a path. `opener_factory(hosts)` returns the `open_stream` for
    `download_verified` (default: `cuda_pack.make_opener`, https only, redirects only between `hosts`)."""

    def __init__(self, home: Path, catalog: dict[str, Model] | None = None, *,
                 in_use: Callable[[], set[str]] = lambda: set(),
                 opener_factory: Callable | None = None,
                 free_space: Callable[[Path], int] = free_bytes,
                 observer: Callable[[str, int, int], None] | None = None,
                 _allow_http_localhost: bool = False) -> None:
        self.home = home
        self._observer = observer  # (file name, done, total) per file, like download.set_observer (the tray tooltip)
        self.catalog = CATALOG if catalog is None else catalog
        self._in_use = in_use
        self._allow_http = _allow_http_localhost  # test hook only; never set from config or env
        self._factory = opener_factory or (
            lambda hosts: cuda_pack.make_opener(hosts, _allow_http_localhost=self._allow_http))
        self._free = free_space
        self._lock = threading.Lock()
        self._state = DownloadState()
        self._busy = False
        self._thread: threading.Thread | None = None
        self._cancel = threading.Event()

    @property
    def state(self) -> DownloadState:
        with self._lock:
            return self._state

    def _set(self, **kw) -> None:
        with self._lock:
            self._state = DownloadState(**{**self._state.__dict__, **kw})

    def status(self, m: Model) -> Status:
        st = self.state
        if self._busy and st.model_id == m.id and st.phase in ("downloading", "verifying"):
            return Status("downloading", st.done)
        d = model_dir(self.home, m)
        if m.files:
            sizes = [_size(d / f.name) for f in m.files]
            ok = all(n == f.size for n, f in zip(sizes, m.files))
        else:  # PENDING_PIN: judge by the expected file names only
            sizes = [_size(d / n) for n in m.probe]
            ok = all(n is not None for n in sizes)
        if ok:
            return Status("installed", sum(n or 0 for n in sizes))
        part = sum(_size(d / (n + ".part")) or 0 for n in m.names)
        if part:
            return Status("partial", part)
        if m.kind == "stt" and any((self.home / "models" / "whisper").glob(f"models--*--faster-whisper-{m.key}")):
            return Status("legacy", 0)  # an older download made by faster-whisper itself (not verified)
        return Status("not_installed", 0)

    def scan(self) -> dict[str, Status]:
        return {mid: self.status(m) for mid, m in self.catalog.items()}

    def download(self, model_id: str) -> None:
        """Start downloading a catalog model in a worker thread. Raises ModelError for anything not allowed (unknown id,
        no pin, already installed, another download running, not enough disk space)."""
        m = self.catalog.get(model_id) if isinstance(model_id, str) else None
        if m is None:
            raise ModelError("Parley does not know that model.")
        if not m.pinned:
            raise ModelError(f"Parley has no verified copy of {m.label} yet, so it will not download it.")
        validate_model(m, allow_http_localhost=self._allow_http)
        d = _confined(self.home, model_dir(self.home, m))
        if self.status(m).state == "installed":
            raise ModelError(f"{m.label} is already installed.")
        have = sum(_size(d / (f.name + ".part")) or 0 for f in m.files)
        need = m.total_bytes - have + FREE_MARGIN
        try:
            free = self._free(d)
        except OSError:
            free = 0
        if free < need:
            raise ModelError(f"Not enough free disk space for {m.label}: it needs {need / 1024**3:.1f} GB free.")
        with self._lock:
            if self._busy:
                raise ModelError("Another model is already downloading. Wait for it or cancel it first.")
            self._busy = True
            self._cancel = threading.Event()
            self._state = DownloadState(m.id, 0, len(m.files), 0, m.total_bytes, "downloading")
            self._thread = threading.Thread(target=self._run, args=(m, self._cancel), daemon=True, name="model-download")
            self._thread.start()

    def cancel(self) -> None:
        """Stop the running download; the `.part` file is kept so a later download continues."""
        self._cancel.set()

    def wait(self, timeout: float | None = None) -> None:
        t = self._thread
        if t is not None:
            t.join(timeout)

    def _run(self, m: Model, cancel: threading.Event) -> None:
        try:
            self._fetch_all(m, cancel)
            self._set(phase="done", done=m.total_bytes, message=f"{m.label} is ready.")
        except DownloadCancelled:
            self._set(phase="cancelled", message="Download cancelled. What was downloaded is kept so you can continue.")
        except ModelError as e:
            self._set(phase="error", message=str(e))
        except BaseException as e:  # noqa: BLE001  (a worker must never die silently)
            log.warning("Model download failed: %s", type(e).__name__)
            self._set(phase="error", message=_explain(e, m.label))
        finally:
            with self._lock:
                self._busy = False

    def _notify(self, name: str, done: int, total: int) -> None:
        if self._observer is not None:
            try:
                self._observer(name, done, total)
            except Exception:  # a broken display must never fail a download
                log.debug("download observer failed", exc_info=True)

    def _fetch_all(self, m: Model, cancel: threading.Event) -> None:
        d = _confined(self.home, model_dir(self.home, m))
        opener = self._factory(m.hosts)
        base = 0
        for i, f in enumerate(m.files):
            dest = _confined(self.home, d / f.name)
            self._set(file_i=i, phase="downloading")
            if _size(dest) == f.size and _sha256_file(dest) == f.sha256:  # e.g. Kokoro's voices file, shared
                base += f.size
                self._set(done=base)
                continue
            dest.unlink(missing_ok=True)  # a same-named file that is not the pinned one is never trusted

            def progress(done: int, total: int, base: int = base, name: str = f.name) -> None:
                self._set(done=base + done, phase="verifying" if done >= total else "downloading")
                self._notify(name, done, total)
            download_verified(opener, f.url, dest, f.sha256, f.size, progress, cancel, resume=True)
            self._notify(f.name, f.size, f.size)  # the end of a file, as download() reports it
            base += f.size

    def delete(self, model_id: str) -> int:
        """Remove a catalog model's files and leftovers; returns the bytes freed. Refuses the model in use, one that is
        downloading, files other installed models still need, anything outside <home>/models, and links."""
        m = self.catalog.get(model_id) if isinstance(model_id, str) else None
        if m is None:
            raise ModelError("Parley can only delete models from its own list.")
        if model_id in self._in_use():
            raise ModelError(f"{m.label} is in use. Switch to another model first, then delete it.")
        if self._busy and self.state.model_id == m.id:
            raise ModelError(f"{m.label} is downloading. Cancel the download first.")
        d = _confined(self.home, model_dir(self.home, m))
        keep: set[str] = set()
        if not m.subdir:  # Kokoro models share voices-v1.0.bin: keep what another installed model needs
            for o in self.catalog.values():
                if o.id != m.id and o.kind == m.kind and self.status(o).state == "installed":
                    keep |= set(o.names)
        freed = 0
        try:
            if m.subdir:  # one whitelisted folder (a Whisper model)
                if d.exists():
                    for p in d.rglob("*"):
                        _confined(self.home, p)
                        freed += _size(p) or 0
                    shutil.rmtree(d)
            else:
                for name in m.names:
                    for target in (d / name, d / (name + ".part")):
                        if target.name in keep:
                            continue
                        _confined(self.home, target)
                        n = _size(target)
                        if n is not None:
                            target.unlink()
                            freed += n
        except PermissionError as e:
            raise ModelError(f"{m.label} is open in another program. Close it or switch models, then try again.") from e
        except OSError as e:
            raise ModelError(f"{m.label} could not be deleted ({e.strerror or 'file error'}).") from e
        return freed


def download_blocking(m: Model, home: Path, open_stream=None) -> None:
    """Download one catalog model and wait (the startup path of `llama_server.ensure_model`). Same checks as the manager."""
    mgr = ModelManager(home, {m.id: m}, opener_factory=(lambda hosts: open_stream) if open_stream else None,
                       observer=download._observer)  # first-run progress for the tray tooltip
    mgr.download(m.id)
    mgr.wait()
    st = mgr.state
    if st.phase != "done":
        raise ModelError(st.message or "The download did not finish.")
