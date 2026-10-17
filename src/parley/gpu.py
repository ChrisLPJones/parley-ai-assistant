"""GPU detection and the pip-installed CUDA libraries, so no CUDA toolkit install is ever needed."""
import functools
import importlib.util
import logging
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass

log = logging.getLogger(__name__)

NVIDIA_PACKAGES = ("nvidia.cublas", "nvidia.cudnn", "nvidia.cuda_runtime", "nvidia.cufft")


@dataclass
class NvidiaGpu:
    name: str
    vram_mb: int
    driver_major: int


@functools.cache
def nvidia() -> NvidiaGpu | None:
    """The first NVIDIA GPU, via nvidia-smi (installed with the driver), or None."""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        out = subprocess.run(
            [exe, "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        ).stdout.strip().splitlines()
        name, vram, driver = (s.strip() for s in out[0].split(","))
        return NvidiaGpu(name, int(float(vram)), int(driver.split(".")[0]))
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return None


@functools.cache
def add_cuda_dll_dirs() -> None:
    """Put the CUDA 12 runtime, cuBLAS, cuFFT and cuDNN from the nvidia-* wheels on the DLL search
    path. PATH is needed as well as add_dll_directory because cuDNN loads its sub-libraries by name."""
    for pkg in NVIDIA_PACKAGES:
        try:
            spec = importlib.util.find_spec(pkg)
        except ModuleNotFoundError:
            continue
        for loc in (spec.submodule_search_locations or []) if spec else []:
            lib_dir = os.path.join(loc, "bin" if sys.platform == "win32" else "lib")
            if not os.path.isdir(lib_dir):
                continue
            if sys.platform == "win32":
                os.add_dll_directory(lib_dir)
            os.environ["PATH"] = lib_dir + os.pathsep + os.environ.get("PATH", "")
