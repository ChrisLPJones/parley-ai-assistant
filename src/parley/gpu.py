"""GPU detection and the pip-installed CUDA libraries, so no CUDA toolkit install is ever needed."""
import functools
import importlib.util
import logging
import os
import platform
import re
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


@dataclass
class Adapter:
    """One graphics adapter as Windows reports it (DXGI)."""
    name: str
    vendor: str  # nvidia | amd | intel | other
    dedicated_mb: int  # own video memory; an integrated GPU reports only a small carve-out here
    shared_mb: int  # system memory the GPU may borrow


VENDORS = {0x10DE: "nvidia", 0x1002: "amd", 0x8086: "intel"}
DISCRETE_MIN_MB = 4096  # AMD/Intel adapters with less dedicated memory are treated as integrated


def _dxgi_adapters() -> list[Adapter]:
    """Enumerate adapters through DXGI with ctypes: no dependency, no subprocess. Windows only."""
    import ctypes
    from ctypes import wintypes

    class GUID(ctypes.Structure):
        _fields_ = [("a", wintypes.DWORD), ("b", wintypes.WORD), ("c", wintypes.WORD), ("d", ctypes.c_ubyte * 8)]

    class DESC1(ctypes.Structure):
        _fields_ = [("Description", ctypes.c_wchar * 128), ("VendorId", wintypes.UINT), ("DeviceId", wintypes.UINT),
                    ("SubSysId", wintypes.UINT), ("Revision", wintypes.UINT), ("DedicatedVideoMemory", ctypes.c_size_t),
                    ("DedicatedSystemMemory", ctypes.c_size_t), ("SharedSystemMemory", ctypes.c_size_t),
                    ("LuidLow", wintypes.DWORD), ("LuidHigh", wintypes.LONG), ("Flags", wintypes.UINT)]

    def method(obj, index, *argtypes):
        """Call slot `index` of a COM object's vtable (HRESULT result)."""
        vtable = ctypes.cast(obj, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
        return ctypes.WINFUNCTYPE(ctypes.HRESULT, ctypes.c_void_p, *argtypes)(vtable[index])

    # IID_IDXGIFactory1 = 770aae78-f26f-4dba-a829-253c83d1b387
    iid = GUID(0x770AAE78, 0xF26F, 0x4DBA, (ctypes.c_ubyte * 8)(0xA8, 0x29, 0x25, 0x3C, 0x83, 0xD1, 0xB3, 0x87))
    factory = ctypes.c_void_p()
    ctypes.WinDLL("dxgi").CreateDXGIFactory1(ctypes.byref(iid), ctypes.byref(factory))
    found: list[Adapter] = []
    try:
        for i in range(16):
            adapter = ctypes.c_void_p()
            try:
                method(factory, 12, wintypes.UINT, ctypes.POINTER(ctypes.c_void_p))(factory, i, ctypes.byref(adapter))  # EnumAdapters1
            except OSError:
                break  # DXGI_ERROR_NOT_FOUND: no more adapters
            try:
                desc = DESC1()
                method(adapter, 10, ctypes.POINTER(DESC1))(adapter, ctypes.byref(desc))  # GetDesc1
                if desc.Flags & 2:  # DXGI_ADAPTER_FLAG_SOFTWARE: "Microsoft Basic Render Driver"
                    continue
                found.append(Adapter(desc.Description.strip(), VENDORS.get(desc.VendorId, "other"),
                                     desc.DedicatedVideoMemory // 2**20, desc.SharedSystemMemory // 2**20))
            finally:
                method(adapter, 2)(adapter)  # Release
    finally:
        method(factory, 2)(factory)
    return found


@functools.cache
def adapters() -> list[Adapter]:
    """Graphics adapters, or [] when they can't be listed (non-Windows, or any error)."""
    if sys.platform != "win32":
        return []
    try:
        return _dxgi_adapters()
    except Exception as e:  # ctypes/COM failures must never stop Parley starting
        log.debug("Couldn't list graphics adapters: %s", e)
        return []


def cpu_info() -> str:
    return f"{platform.processor() or platform.machine()}, {os.cpu_count()} logical cores"


# What the speech and LLM stages run on. llm: cuda | vulkan | cpu; tts and stt: cuda | cpu.
@dataclass
class Plan:
    llm: str
    tts: str
    stt: str
    preset: str  # small | tiny
    device: str  # human-readable description of the chosen GPU, for logs


# The small preset (Qwen3.5 4B) needs about this much GPU memory including context
SMALL_PRESET_GB = 4
SPEECH_VRAM_MB = 2048  # Whisper large-v3-turbo + Kokoro, loading alongside the LLM on an NVIDIA GPU
# Integrated GPUs share system memory and are often no faster than a modern CPU for token generation
# (the benchmark on the 2-compute-unit Radeon in the dev machine confirmed this), so the LLM stays on
# the CPU unless the user opts in with llm.runtime = "vulkan".
INTEGRATED_LLM = "cpu"


def plan(found: list[Adapter], nvidia_gpu: NvidiaGpu | None) -> Plan:
    """Pick the best way to run each stage from the hardware. Pure, so it is unit tested.

    - NVIDIA: CUDA for everything (the existing behaviour).
    - Discrete AMD/Intel: the LLM on the GPU through Vulkan. Whisper (CTranslate2 has no AMD/Intel
      backend) and Kokoro (its DirectML build fails on the model's ConvTranspose layers) stay on CPU.
    - Integrated GPU only, or no GPU: everything on CPU.
    Intel GPUs follow the AMD path by design but are unverified: no Intel hardware was available."""
    if nvidia_gpu:
        small = (nvidia_gpu.vram_mb - SPEECH_VRAM_MB) / 1024 >= SMALL_PRESET_GB
        return Plan("cuda", "cuda", "cuda", "small" if small else "tiny", nvidia_gpu.name)
    discrete = [a for a in found if a.vendor in ("amd", "intel") and a.dedicated_mb >= DISCRETE_MIN_MB]
    if discrete:
        best = max(discrete, key=lambda a: a.dedicated_mb)
        # Speech runs on the CPU here, so the whole card is for the LLM
        small = best.dedicated_mb / 1024 >= SMALL_PRESET_GB
        return Plan("vulkan", "cpu", "cpu", "small" if small else "tiny", best.name)
    return Plan(INTEGRATED_LLM, "cpu", "cpu", "tiny", "none usable (integrated or no GPU)")


def current_plan() -> Plan:
    return plan(adapters(), nvidia())


def describe() -> str:
    """One line per fact for the startup log; helps with bug reports."""
    gpus = "; ".join(f"{a.name} ({a.vendor}, {a.dedicated_mb} MB dedicated)" for a in adapters()) or "none detected"
    p = current_plan()
    return f"CPU: {cpu_info()}. GPUs: {gpus}. Plan: LLM {p.llm}, speech-to-text {p.stt}, text-to-speech {p.tts}"


def parse_llama_devices(text: str) -> list[tuple[str, str]]:
    """(id, name) pairs from `llama-server --list-devices`, e.g. 'Vulkan0: AMD Radeon(TM) Graphics (2048 MiB, 1900 MiB free)'."""
    return [(m.group(1), m.group(2).strip()) for m in re.finditer(r"^\s*(\w+?\d+): (.+?) \(\d+ MiB", text, re.M)]


def pick_llama_device(devices: list[tuple[str, str]], gpu_name: str) -> str | None:
    """The llama.cpp device id for the chosen adapter, so it doesn't split a model across an iGPU
    and a dGPU. Matches by name; None (omit --device) when not found or ambiguous."""
    def norm(s: str) -> str:
        return re.sub(r"[^a-z0-9]", "", s.lower())

    want = norm(gpu_name)
    hits = [d for d, name in devices if want and (want in norm(name) or norm(name) in want)]
    return hits[0] if len(hits) == 1 else None


def _add_dll_dir(lib_dir: str) -> None:
    if sys.platform == "win32":
        os.add_dll_directory(lib_dir)
    os.environ["PATH"] = lib_dir + os.pathsep + os.environ.get("PATH", "")


@functools.cache
def add_cuda_dll_dirs() -> None:
    """Put the CUDA 12 runtime, cuBLAS, cuFFT and cuDNN on the DLL search path. PATH is needed as well as
    add_dll_directory because cuDNN loads its sub-libraries by name. Source installs use the nvidia-* wheels; a frozen
    copy uses the downloaded CUDA pack (cuda_pack.py) if it is complete, else nothing (the CPU is used)."""
    if getattr(sys, "frozen", False):
        from . import cuda_pack
        from .config import app_dir

        if pack := cuda_pack.usable_dir(app_dir()):
            _add_dll_dir(str(pack))
            log.info("Using the downloaded CUDA pack in %s", pack)
        else:
            log.debug("No complete CUDA pack yet; CUDA libraries are not available")
        return
    for pkg in NVIDIA_PACKAGES:
        try:
            spec = importlib.util.find_spec(pkg)
        except ModuleNotFoundError:
            continue
        for loc in (spec.submodule_search_locations or []) if spec else []:
            lib_dir = os.path.join(loc, "bin" if sys.platform == "win32" else "lib")
            if os.path.isdir(lib_dir):
                _add_dll_dir(lib_dir)
