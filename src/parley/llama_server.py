"""Built-in LLM runtime: downloads a pinned llama.cpp build and a GGUF model, then runs llama-server
as a child process that dies with Parley."""
import atexit
import logging
import platform
import re
import shutil
import socket
import subprocess
import sys
import tarfile
import time
import zipfile
from pathlib import Path

import httpx

from . import cuda_pack, gpu
from .config import LlmConfig
from .download import ArchiveError, DownloadError, download, download_verified, safe_extract_tar, safe_extract_zip

log = logging.getLogger(__name__)

LLAMA_CPP_TAG = "b11702"
RELEASE_URL = f"https://github.com/ggml-org/llama.cpp/releases/download/{LLAMA_CPP_TAG}/"
RELEASE_HOSTS = frozenset({"github.com", "release-assets.githubusercontent.com", "objects.githubusercontent.com"})

# sha256 and size of every release asset `_build_assets` can ask for, from the GitHub release API (`digest`) for
# LLAMA_CPP_TAG; the Windows vulkan-x64 and cuda-13.4-x64 zips, the macOS arm64 and the Ubuntu vulkan-x64 tarballs were
# also downloaded and hashed. Bumping the tag means regenerating this table (tests/test_llama_pins.py checks it is
# complete). An asset that is not listed is never downloaded. The Hugging Face model files are not pinned.
ASSET_PINS: dict[str, tuple[str, int]] = {
    "cudart-llama-b11702-bin-ubuntu-cuda-12.8-x64.tar.gz":
        ("ea7a556a461317708b06e90c436dbf305db93d10939715d6b58aec423df1ad41", 594377730),
    "cudart-llama-b11702-bin-ubuntu-cuda-13.4-x64.tar.gz":
        ("0bc852fd02cce084b983ba78f0ab9d4c7c8a84261b2146909cfec56de526df59", 440236659),
    "cudart-llama-bin-win-cuda-12.4-x64.zip":
        ("8c79a9b226de4b3cacfd1f83d24f962d0773be79f1e7b75c6af4ded7e32ae1d6", 391443627),
    "cudart-llama-bin-win-cuda-13.4-x64.zip":
        ("738f8c251ac22b70c3ae6f83a10cf222725df0395246a2cf58f32bdb85fbe668", 423535356),
    "llama-b11702-bin-macos-arm64.tar.gz":
        ("96d5a4dee4c5e5039eb374aa2465d6942f2c341ca13198c9fb01419b671e1468", 12249372),
    "llama-b11702-bin-macos-x64.tar.gz":
        ("d7fa1b16b3030a3175e12688e2afd88522486f22b25c73465cc808c73185dc2e", 11736034),
    "llama-b11702-bin-ubuntu-cuda-12.8-x64.tar.gz":
        ("78c82f137b92cfc2e77309e7a947afa281b7898fed90c160bc53e08ee7c5c2e4", 172730349),
    "llama-b11702-bin-ubuntu-cuda-13.4-x64.tar.gz":
        ("e89577e8da32ab51c6e75c835eb286cc77f201644d20e3eaa1897516a43e5351", 153839224),
    "llama-b11702-bin-ubuntu-vulkan-arm64.tar.gz":
        ("6a49d48537912db13a48b91d30fe58ee8c02a6cbb0d4025897d0bda535106751", 24250253),
    "llama-b11702-bin-ubuntu-vulkan-x64.tar.gz":
        ("62cdc0ec1b475e4263848168c183a0b1fa7dcb5797668a62be3de22691601233", 30556972),
    "llama-b11702-bin-win-cuda-12.4-x64.zip":
        ("3470f420c53330830ed08b6a0bb79db7199a185c4b6e7a3db0cf865043cb263c", 266833945),
    "llama-b11702-bin-win-cuda-13.4-x64.zip":
        ("39839af853b951bd1454acc458ef7b03d4c8acd6c9da61fa1c29083f011054ea", 154749299),
    "llama-b11702-bin-win-vulkan-arm64.zip":
        ("c9d053cb8517f7ef2b7e553b0a658fcdf637d7647729c9612abb1facc81a3aa7", 24878164),
    "llama-b11702-bin-win-vulkan-x64.zip":
        ("cb34fdf7c628b45adc50954097fba0263a92a9a5413faba1074e790ed8781e14", 32260463),
}

# name: (Hugging Face repo, file, approx memory needed in GB incl. context). Kept small so Parley
# runs on ordinary laptops; bigger GGUFs work via llm.model_repo / model_file.
PRESETS = {
    "small": ("unsloth/Qwen3.5-4B-GGUF", "Qwen3.5-4B-Q4_K_M.gguf", 4),
    "tiny": ("unsloth/Qwen3.5-2B-GGUF", "Qwen3.5-2B-Q4_K_M.gguf", 2),
}
SPEECH_VRAM_MB = gpu.SPEECH_VRAM_MB  # Whisper large-v3-turbo + Kokoro, loading alongside the LLM
RUNTIMES = ("auto", "cuda", "vulkan", "cpu")


def pick_preset() -> str:
    """small on a GPU with room for it (or Apple Silicon), otherwise tiny, which is fine on CPU."""
    if sys.platform == "darwin" and platform.machine() == "arm64":
        return "small"
    return gpu.current_plan().preset


def resolve_runtime(cfg: LlmConfig) -> str:
    """cuda | vulkan | cpu | metal, from llm.runtime or (auto) the hardware plan."""
    if cfg.runtime not in RUNTIMES:
        raise SystemExit(f"Unknown llm.runtime {cfg.runtime!r}; choose one of {', '.join(RUNTIMES)}")
    if cfg.runtime == "cuda" and not gpu.nvidia():
        raise SystemExit('llm.runtime = "cuda" needs an NVIDIA GPU (nvidia-smi was not found)')
    if sys.platform == "darwin" and cfg.runtime != "cpu":
        return "metal"  # the macOS build is Metal; "cpu" still works through --device none
    if cfg.runtime != "auto":
        return cfg.runtime
    if sys.platform != "win32" and not gpu.nvidia():
        return "vulkan"  # adapters can't be listed here; let llama.cpp use whatever Vulkan device it finds
    return gpu.current_plan().llm


def _build_assets(runtime: str) -> tuple[str, list[str]]:
    """(variant label, release asset names) for this machine and runtime. The CPU runtime reuses the
    Vulkan build (it contains the CPU backend) and is started with --device none: no extra download."""
    nv = gpu.nvidia() if runtime == "cuda" else None
    arch = "arm64" if platform.machine().lower() in ("arm64", "aarch64") else "x64"
    if sys.platform == "win32":
        if nv and arch == "x64":
            cuda = "13.4" if nv.driver_major >= 580 else "12.4"
            return f"cuda-{cuda}", [f"llama-{LLAMA_CPP_TAG}-bin-win-cuda-{cuda}-x64.zip", f"cudart-llama-bin-win-cuda-{cuda}-x64.zip"]
        return "vulkan", [f"llama-{LLAMA_CPP_TAG}-bin-win-vulkan-{arch}.zip"]
    if sys.platform == "darwin":
        return "metal", [f"llama-{LLAMA_CPP_TAG}-bin-macos-{arch}.tar.gz"]
    if nv and arch == "x64":
        cuda = "13.4" if nv.driver_major >= 580 else "12.8"
        return f"cuda-{cuda}", [f"llama-{LLAMA_CPP_TAG}-bin-ubuntu-cuda-{cuda}-x64.tar.gz", f"cudart-llama-{LLAMA_CPP_TAG}-bin-ubuntu-cuda-{cuda}-x64.tar.gz"]
    return "vulkan", [f"llama-{LLAMA_CPP_TAG}-bin-ubuntu-vulkan-{arch}.tar.gz"]


class RuntimeDownloadError(RuntimeError):
    """The llama.cpp download is not trusted or not usable; the message is short enough to be spoken."""


def _fetch_asset(asset: str, home: Path, open_stream=None) -> Path:
    """Download one pinned release asset (fresh, size and sha256 checked, deleted on mismatch). Unknown assets are refused."""
    pin = ASSET_PINS.get(asset)
    if pin is None:
        raise RuntimeDownloadError("Parley has no verified copy of the language model runtime for this computer, so it will not download one.")
    dest = home / "downloads" / asset
    dest.unlink(missing_ok=True)  # never trust a leftover file; a verified fetch always starts fresh
    try:
        return download_verified(open_stream or cuda_pack.make_opener(RELEASE_HOSTS), RELEASE_URL + asset, dest, *pin)
    except (DownloadError, cuda_pack.PackError) as e:
        raise RuntimeDownloadError(f"The language model runtime download failed its safety check ({e}), so Parley discarded it.") from e


def ensure_runtime(home: Path, runtime: str, open_stream=None) -> Path:
    """Download (pinned and verified) and unpack llama.cpp once; returns the llama-server executable."""
    variant, assets = _build_assets(runtime)
    target = home / "runtime" / f"llama.cpp-{LLAMA_CPP_TAG}-{variant}"
    exe_name = "llama-server.exe" if sys.platform == "win32" else "llama-server"
    if not (target / ".complete").exists():
        log.info("Downloading llama.cpp %s (%s), first run only ...", LLAMA_CPP_TAG, variant)
        shutil.rmtree(target, ignore_errors=True)  # a half-extracted earlier attempt
        target.mkdir(parents=True, exist_ok=True)
        for asset in assets:
            archive = _fetch_asset(asset, home, open_stream)
            try:
                if asset.endswith(".zip"):
                    safe_extract_zip(archive, target)
                else:
                    safe_extract_tar(archive, target)
            except ArchiveError as e:
                shutil.rmtree(target, ignore_errors=True)
                raise RuntimeDownloadError(f"The language model runtime archive looked unsafe ({e}), so Parley did not unpack it.") from e
            except (zipfile.BadZipFile, tarfile.TarError, OSError) as e:  # corrupt archive, full disk, locked file
                log.warning("Unpacking the llama.cpp runtime failed: %r", e)
                shutil.rmtree(target, ignore_errors=True)
                raise RuntimeDownloadError("The language model runtime could not be unpacked. Check the free disk space and try again.") from e
            finally:
                archive.unlink(missing_ok=True)
        (target / ".complete").touch()
    exe = next(target.rglob(exe_name), None)
    if not exe:
        raise RuntimeError(f"{exe_name} not found in {target}")
    return exe


MODEL_REPO_RE = re.compile(r"[\w.-]+/[\w.-]+", re.ASCII)
MODEL_FILE_RE = re.compile(r"[\w.-]+\.gguf", re.ASCII)
_WIN_DEVICES = frozenset({"con", "prn", "aux", "nul", *(f"com{i}" for i in range(1, 10)), *(f"lpt{i}" for i in range(1, 10))})


def check_model_names(repo: str, file: str) -> None:
    """Raise ValueError unless repo and file are plain names: the file becomes a path under models/llm and the
    repo part of a Hugging Face URL, so no separators, drive letters, '..' or Windows device names."""
    if not MODEL_REPO_RE.fullmatch(repo) or any(p in (".", "..") for p in repo.split("/")):
        raise ValueError("a Hugging Face repo looks like owner/name (letters, digits, '.', '_' and '-' only)")
    if not MODEL_FILE_RE.fullmatch(file) or file.split(".")[0].lower() in _WIN_DEVICES:
        raise ValueError("a model file is a plain name ending in .gguf (letters, digits, '.', '_' and '-' only)")


def ensure_model(cfg: LlmConfig, home: Path) -> Path:
    if cfg.model_path:
        return Path(cfg.model_path)
    if cfg.model_repo:
        repo, file = cfg.model_repo, cfg.model_file
        try:
            check_model_names(repo, file)
        except ValueError as e:
            raise SystemExit(f"Invalid llm.model_repo / llm.model_file: {e}.") from e
    else:
        preset = cfg.preset if cfg.preset != "auto" else pick_preset()
        if preset not in PRESETS:
            raise SystemExit(f"Unknown llm.preset {preset!r}; choose auto or one of {', '.join(PRESETS)}")
        repo, file, _ = PRESETS[preset]
        log.info("LLM preset: %s (%s)", preset, file)
    base = home / "models" / "llm"
    dest = base / file
    if not dest.resolve().is_relative_to(base.resolve()):
        raise SystemExit("llm.model_file must stay inside the models folder.")
    if not dest.exists():
        log.info("Downloading %s from %s, first run only ...", file, repo)
        download(f"https://huggingface.co/{repo}/resolve/main/{file}", dest)
    return dest


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _kill_with_parent(proc: subprocess.Popen) -> None:
    """On Windows, put the server in a job object so it's killed even if Parley crashes."""
    atexit.register(proc.kill)
    if sys.platform != "win32":
        return
    import ctypes
    from ctypes import wintypes

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.CreateJobObjectW.restype = wintypes.HANDLE
    k32.OpenProcess.restype = wintypes.HANDLE

    class LIMIT_INFO(ctypes.Structure):
        _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
                    ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                    ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                    ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD), ("SchedulingClass", wintypes.DWORD)]

    class EXTENDED_INFO(ctypes.Structure):
        _fields_ = [("BasicLimitInformation", LIMIT_INFO), ("IoInfo", ctypes.c_uint64 * 6),
                    ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
                    ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t)]

    job = k32.CreateJobObjectW(None, None)
    info = EXTENDED_INFO()
    info.BasicLimitInformation.LimitFlags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    k32.SetInformationJobObject(job, 9, ctypes.byref(info), ctypes.sizeof(info))  # JobObjectExtendedLimitInformation
    handle = k32.OpenProcess(0x0101, False, proc.pid)  # PROCESS_SET_QUOTA | PROCESS_TERMINATE
    if not k32.AssignProcessToJobObject(job, handle):
        log.debug("Couldn't attach llama-server to a job object (error %d)", ctypes.get_last_error())
    k32.CloseHandle(handle)
    _kill_with_parent.job = job  # handle must stay open for Parley's lifetime


def _vulkan_device(exe: Path) -> str | None:
    """The `--device` id of the best AMD/Intel adapter, so llama.cpp doesn't split the model across
    an integrated and a discrete GPU. None (flag omitted) when it can't be told."""
    found = [a for a in gpu.adapters() if a.vendor in ("amd", "intel")] or gpu.adapters()
    if not found:
        return None
    best = max(found, key=lambda a: a.dedicated_mb)
    try:
        out = subprocess.run([str(exe), "--list-devices"], capture_output=True, text=True, timeout=60, cwd=exe.parent,
                             creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except (OSError, subprocess.SubprocessError):
        return None
    devices = gpu.parse_llama_devices(out.stdout + out.stderr)
    device = gpu.pick_llama_device([d for d in devices if d[0].lower().startswith("vulkan")], best.name)
    log.debug("llama.cpp devices %s; using %s for %s", devices, device, best.name)
    return device


def _launch(exe: Path, model: Path, cfg: LlmConfig, runtime: str, log_path: Path) -> tuple[subprocess.Popen, str]:
    port = _free_port()
    args = [
        str(exe), "--model", str(model), "--host", "127.0.0.1", "--port", str(port),
        "--ctx-size", str(cfg.num_ctx), "--jinja", "--no-webui",
    ]
    if runtime == "cpu":
        args += ["--device", "none", "--n-gpu-layers", "0"]
    else:
        if runtime == "vulkan" and (device := _vulkan_device(exe)):
            args += ["--device", device]
        if cfg.gpu_layers >= 0:
            args += ["--n-gpu-layers", str(cfg.gpu_layers)]
        else:
            # Offload as many layers as fit, leaving room for the speech models that load at the same time
            args += ["--fit", "on", "--fit-target", str(SPEECH_VRAM_MB)]
    if not cfg.think:
        args += ["--reasoning-budget", "0"]
    with open(log_path, "wb") as log_file:
        proc = subprocess.Popen(
            args, stdout=log_file, stderr=subprocess.STDOUT, cwd=exe.parent,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    _kill_with_parent(proc)
    return proc, f"http://127.0.0.1:{port}"


def _wait_ready(proc: subprocess.Popen, url: str, log_path: Path) -> None:
    while True:
        if proc.poll() is not None:
            tail = log_path.read_text(errors="replace").strip().splitlines()[-15:]
            raise RuntimeError("llama-server exited during startup:\n  " + "\n  ".join(tail))
        try:
            if httpx.get(url + "/health", timeout=2).status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(0.25)


def start(cfg: LlmConfig, home: Path) -> tuple[str, str]:
    """Start llama-server and wait until the model is loaded. Returns (OpenAI-style base URL, model name)."""
    runtime = resolve_runtime(cfg)
    exe = ensure_runtime(home, runtime)
    model = ensure_model(cfg, home)
    log_path = home / "logs" / "llama-server.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    start_time = time.perf_counter()
    proc, url = _launch(exe, model, cfg, runtime, log_path)
    try:
        _wait_ready(proc, url, log_path)
    except RuntimeError as e:
        if runtime in ("cpu", "metal"):
            raise
        # A GPU runtime that won't start (old driver, no Vulkan device) must not leave Parley without a brain
        # The retry overwrites llama-server.log, so keep the failed run's tail in Parley's own log
        log.warning("LLM on %s failed to start; retrying on the CPU. %s", runtime, e)
        runtime = "cpu"
        proc, url = _launch(exe, model, cfg, runtime, log_path)
        _wait_ready(proc, url, log_path)
    log.info("LLM server up: %s on %s (%.1fs, log %s)", model.name, runtime, time.perf_counter() - start_time, log_path)
    return url + "/v1", model.stem
