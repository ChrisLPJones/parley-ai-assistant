"""Plain-language text for the Advanced page's device settings: which hardware a choice would run on right now.

Pure (the probe is injected), so it is tested with fake hardware. Never mentions drivers or driver versions.
"""
import os
import platform
import sys
from dataclasses import dataclass, field

from . import gpu

STAGES = {"stt": "Speech recognition", "tts": "The voice", "llm": "The language model"}


@dataclass
class Facts:
    """What this computer has: the NVIDIA card (if any), every graphics adapter, the processor."""
    nvidia: gpu.NvidiaGpu | None = None
    adapters: list[gpu.Adapter] = field(default_factory=list)
    cpu: str = "Unknown processor"
    cores: int = 0


def cpu_name() -> str:
    if sys.platform == "win32":
        try:
            import winreg

            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as k:
                name = str(winreg.QueryValueEx(k, "ProcessorNameString")[0]).strip()
            if name:
                return " ".join(name.split())
        except OSError:
            pass
    return platform.processor() or platform.machine() or "Unknown processor"


def probe() -> Facts:
    """Ask this computer (slow the first time: worker thread only)."""
    return Facts(gpu.nvidia(), list(gpu.adapters()), cpu_name(), os.cpu_count() or 0)


def _gb(mb: int) -> str:
    return f"{max(1, round(mb / 1024))} GB"


def _cpu(f: Facts) -> str:
    return f"CPU: {f.cpu}" + (f", {f.cores} logical cores" if f.cores else "")


def _nvidia(f: Facts) -> str:
    return f"{f.nvidia.name}, {_gb(f.nvidia.vram_mb)}"


def _best_adapter(f: Facts) -> gpu.Adapter | None:
    real = [a for a in f.adapters if a.vendor != "nvidia" or f.nvidia is None]
    return max(real, key=lambda a: a.dedicated_mb, default=None)


def _adapter(a: gpu.Adapter) -> str:
    if a.dedicated_mb >= gpu.DISCRETE_MIN_MB:
        return f"{a.name}, {_gb(a.dedicated_mb)}"
    return f"{a.name} (integrated: it shares the computer's memory and is often no faster than the CPU)"


def device_line(stage: str, choice: str, f: Facts) -> str:
    """One or two sentences: what `choice` (auto / cuda / vulkan / cpu) would run on for `stage` (stt / tts / llm)."""
    plan = gpu.plan(f.adapters, f.nvidia)
    if choice == "auto":
        if getattr(plan, stage) == "cuda":
            where = _nvidia(f)
        elif getattr(plan, stage) == "vulkan":
            best = _best_adapter(f)
            where = _adapter(best) if best else _cpu(f)
        else:
            where = _cpu(f)
        note = ""
        if stage != "llm" and f.nvidia is None and any(a.vendor in ("amd", "intel") for a in f.adapters):
            note = " Speech can only use an NVIDIA graphics card, so your graphics card is not used for it."
        return f"Runs on: {where}.{note}"
    if choice == "cpu":
        return f"Runs on: {_cpu(f)}."
    if choice == "cuda":
        if f.nvidia is not None:
            return f"Runs on: {_nvidia(f)}."
        if stage == "llm":
            return "Not available on this PC: no NVIDIA graphics card was found. The language model would not start with this choice."
        return f"Not available on this PC: no NVIDIA graphics card was found. Parley will use the CPU instead ({_cpu(f)})."
    if choice == "vulkan":
        best = _best_adapter(f)
        if f.nvidia is not None and (best is None or best.dedicated_mb < f.nvidia.vram_mb):
            return f"Runs on: {_nvidia(f)}."
        if best is not None:
            return f"Runs on: {_adapter(best)}."
        return f"Not available on this PC: no graphics card was found. Parley will use the CPU instead ({_cpu(f)})."
    return ""


# One plain sentence per option, for the dropdowns whose names are jargon.
DEVICE_OPTIONS = {
    "auto": "Auto (recommended): uses an NVIDIA graphics card when there is one, otherwise the CPU.",
    "cuda": "Cuda: NVIDIA's graphics-card mode. Fast, but only works with an NVIDIA graphics card.",
    "cpu": "Cpu: the main processor. Works on every computer, but slower than a graphics card.",
}
COMPUTE_OPTIONS = {
    "default": "Default (recommended): Whisper picks the best precision your NVIDIA card supports.",
    "float16": "Float16: half-precision numbers. Full accuracy and fast on recent NVIDIA cards, but uses more graphics memory.",
    "int8_float16": "Int8 float16: compact numbers with half-precision maths. Much less graphics memory for a tiny loss of accuracy; "
                    "suits cards with little memory.",
    "int8": "Int8: the smallest numbers. Least graphics memory and can be a little less accurate; suits small or older cards.",
}
RUNTIME_OPTIONS = {
    "auto": "Auto (recommended): CUDA on an NVIDIA card, Vulkan on a separate AMD or Intel graphics card, otherwise the CPU.",
    "cuda": "Cuda: NVIDIA's graphics-card mode. Usually the fastest, needs an NVIDIA graphics card.",
    "vulkan": "Vulkan: a graphics mode that works on AMD, Intel and NVIDIA cards (Intel cards are untested).",
    "cpu": "Cpu: the main processor only. Works everywhere but is slower and uses ordinary memory instead of graphics memory.",
}
OPTION_TEXT = {"stt.device": DEVICE_OPTIONS, "tts.device": DEVICE_OPTIONS, "stt.compute_type": COMPUTE_OPTIONS,
               "llm.runtime": RUNTIME_OPTIONS}
STAGE_OF = {"stt.device": "stt", "tts.device": "tts", "llm.runtime": "llm"}
