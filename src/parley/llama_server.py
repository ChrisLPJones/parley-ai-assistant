"""Built-in LLM runtime: downloads a pinned llama.cpp build and a GGUF model, then runs llama-server
as a child process that dies with Parley."""
import atexit
import logging
import platform
import socket
import subprocess
import sys
import tarfile
import time
import zipfile
from pathlib import Path

import httpx

from . import gpu
from .config import LlmConfig
from .download import download

log = logging.getLogger(__name__)

LLAMA_CPP_TAG = "b11702"
RELEASE_URL = f"https://github.com/ggml-org/llama.cpp/releases/download/{LLAMA_CPP_TAG}/"

# name: (Hugging Face repo, file, approx memory needed in GB incl. context). Kept small so Parley
# runs on ordinary laptops; bigger GGUFs work via llm.model_repo / model_file.
PRESETS = {
    "small": ("unsloth/Qwen3.5-4B-GGUF", "Qwen3.5-4B-Q4_K_M.gguf", 4),
    "tiny": ("unsloth/Qwen3.5-2B-GGUF", "Qwen3.5-2B-Q4_K_M.gguf", 2),
}
SPEECH_VRAM_MB = 2048  # Whisper large-v3-turbo + Kokoro, loading alongside the LLM


def pick_preset() -> str:
    """small on a GPU with room for it (or Apple Silicon), otherwise tiny, which is fine on CPU."""
    if sys.platform == "darwin" and platform.machine() == "arm64":
        return "small"
    nv = gpu.nvidia()
    if nv and (nv.vram_mb - SPEECH_VRAM_MB) / 1024 >= PRESETS["small"][2]:
        return "small"
    return "tiny"


def _build_assets() -> tuple[str, list[str]]:
    """(variant label, release asset names) for this machine."""
    nv = gpu.nvidia()
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


def ensure_runtime(home: Path) -> Path:
    """Download and unpack llama.cpp once; returns the llama-server executable."""
    variant, assets = _build_assets()
    target = home / "runtime" / f"llama.cpp-{LLAMA_CPP_TAG}-{variant}"
    exe_name = "llama-server.exe" if sys.platform == "win32" else "llama-server"
    if not (target / ".complete").exists():
        log.info("Downloading llama.cpp %s (%s), first run only ...", LLAMA_CPP_TAG, variant)
        target.mkdir(parents=True, exist_ok=True)
        for asset in assets:
            archive = download(RELEASE_URL + asset, home / "downloads" / asset)
            if asset.endswith(".zip"):
                with zipfile.ZipFile(archive) as z:
                    z.extractall(target)
            else:
                with tarfile.open(archive) as t:
                    t.extractall(target, filter="data")
            archive.unlink()
        (target / ".complete").touch()
    exe = next(target.rglob(exe_name), None)
    if not exe:
        raise RuntimeError(f"{exe_name} not found in {target}")
    return exe


def ensure_model(cfg: LlmConfig, home: Path) -> Path:
    if cfg.model_path:
        return Path(cfg.model_path)
    if cfg.model_repo:
        repo, file = cfg.model_repo, cfg.model_file
    else:
        preset = cfg.preset if cfg.preset != "auto" else pick_preset()
        if preset not in PRESETS:
            raise SystemExit(f"Unknown llm.preset {preset!r}; choose auto or one of {', '.join(PRESETS)}")
        repo, file, _ = PRESETS[preset]
        log.info("LLM preset: %s (%s)", preset, file)
    dest = home / "models" / "llm" / file
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


def start(cfg: LlmConfig, home: Path) -> tuple[str, str]:
    """Start llama-server and wait until the model is loaded. Returns (OpenAI-style base URL, model name)."""
    exe = ensure_runtime(home)
    model = ensure_model(cfg, home)
    port = _free_port()
    args = [
        str(exe), "--model", str(model), "--host", "127.0.0.1", "--port", str(port),
        "--ctx-size", str(cfg.num_ctx), "--jinja", "--no-webui",
    ]
    if cfg.gpu_layers >= 0:
        args += ["--n-gpu-layers", str(cfg.gpu_layers)]
    else:
        # Offload as many layers as fit, leaving room for the speech models that load at the same time
        args += ["--fit", "on", "--fit-target", str(SPEECH_VRAM_MB)]
    if not cfg.think:
        args += ["--reasoning-budget", "0"]
    log_path = home / "logs" / "llama-server.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    start_time = time.perf_counter()
    with open(log_path, "wb") as log_file:
        proc = subprocess.Popen(
            args, stdout=log_file, stderr=subprocess.STDOUT, cwd=exe.parent,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    _kill_with_parent(proc)
    url = f"http://127.0.0.1:{port}"
    while True:
        if proc.poll() is not None:
            tail = log_path.read_text(errors="replace").strip().splitlines()[-15:]
            raise RuntimeError("llama-server exited during startup:\n  " + "\n  ".join(tail))
        try:
            if httpx.get(url + "/health", timeout=2).status_code == 200:
                break
        except httpx.HTTPError:
            pass
        time.sleep(0.25)
    log.info("LLM server up: %s (%.1fs, log %s)", model.name, time.perf_counter() - start_time, log_path)
    return url + "/v1", model.stem
