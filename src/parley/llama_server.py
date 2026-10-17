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


def ensure_runtime(home: Path, runtime: str) -> Path:
    """Download and unpack llama.cpp once; returns the llama-server executable."""
    variant, assets = _build_assets(runtime)
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
