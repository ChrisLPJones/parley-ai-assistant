"""Plain-text formatting and button rules for the Settings "Models" page (no Tk, so it is tested without a display)."""
from dataclasses import dataclass

from . import models

GROUPS = (("stt", "Speech recognition (Whisper)"), ("llm", "Language model"), ("tts", "Voice (Kokoro)"))
LIVE_KINDS = frozenset({"llm"})  # kinds whose Download / Use / Delete work today; the rest are listed read-only
COMING_SOON = "Choosing and downloading this kind of model is coming in the next update."
LLM_RESTART_NOTE = "Press Save to choose it. The language model changes after restarting Parley."
OWN_REPLACE_NOTE = "This replaces your own model setting when you press Save."
OWN_ID = "own-file"
SYMBOLS = {"good": "✓", "slow": "!", "no": "✗"}  # check, exclamation, cross: the word is always shown too


def ordered(kind: str, catalog) -> list[models.Model]:
    """The models of one kind, largest download first; equal sizes keep catalog order (sorted() is stable)."""
    return sorted((m for m in catalog.values() if m.kind == kind), key=lambda m: -m.total_bytes)


def format_size(n: int) -> str:
    if n <= 0:
        return "-"
    if n >= 1024**3:
        return f"{n / 1024**3:.1f} GB"
    return f"{max(1, round(n / 1024**2))} MB"


def hardware_line(hw: models.Hardware) -> str:
    parts = []
    if hw.gpu_name:
        parts.append(f"{hw.gpu_name} ({max(1, round(hw.vram_mb / 1024))} GB video memory)")
    else:
        parts.append("No supported graphics card found")
    if hw.ram_mb:
        parts.append(f"{max(1, round(hw.ram_mb / 1024))} GB memory")
    if hw.free_disk_mb:
        parts.append(f"{hw.free_disk_mb / 1024:.0f} GB free on this drive")
    return ", ".join(parts)


def fit_text(level: str, words: str) -> str:
    return f"{SYMBOLS.get(level, '')} {words}".strip()


def percent(done: int, total: int) -> int:
    return min(100, done * 100 // total) if total > 0 else 0


def progress_text(st: models.DownloadState) -> str:
    if st.phase == "verifying":
        return "Checking the download..."
    gb = lambda n: f"{n / 1024**3:.1f}"  # noqa: E731
    return f"{gb(st.done)} of {gb(st.total)} GB ({percent(st.done, st.total)}%)"


def warning_text(level: str) -> str:
    """Said next to the fit verdict for models that may not run well. Never blocks: the user decides."""
    if level == "slow":
        return "You can still use it, but expect slower answers."
    if level == "no":
        return "It may not run on this computer. You can still try it, at your own risk."
    return ""


def status_text(m: models.Model, st: models.Status, in_use: bool, chosen: str = "") -> str:
    """chosen: "" | "unsaved" (Use pressed, Save not yet) | "saved" (written to config.toml, needs a restart)."""
    if st.state == "downloading":
        return f"Downloading {percent(st.bytes, m.total_bytes)}%"
    if in_use:
        base = "In use"
    elif chosen:
        base = "Chosen (press Save)" if chosen == "unsaved" else "Chosen (restart Parley)"
    elif st.state == "installed":
        base = "Installed"
    elif st.state == "partial":
        base = f"Partly ({format_size(st.bytes)})"
    elif st.state == "legacy":
        base = "Installed (older download)"
    elif not m.pinned:
        base = "Not available yet"
    else:
        base = "Not installed"
    return base


@dataclass(frozen=True)
class Actions:
    download: bool
    cancel: bool
    use: bool
    delete: bool
    note: str = ""  # why something is off, shown under the buttons


def actions(m: models.Model | None, st: models.Status | None, *, in_use: bool, selected: bool, other_chosen: bool,
            dl: models.DownloadState) -> Actions:
    """Which buttons are on for the selected row. One download at a time; nothing for kinds not yet supported."""
    if m is None or st is None:
        return Actions(False, False, False, False)
    running = dl.phase in ("downloading", "verifying")
    mine = running and dl.model_id == m.id
    if m.kind not in LIVE_KINDS:
        return Actions(False, False, False, False, COMING_SOON)
    installed = st.state == "installed"
    can_download = m.pinned and st.state in ("not_installed", "partial", "legacy") and not running
    note = ""
    if not m.pinned and not installed:
        note = "Parley has no verified copy of this model yet."
    elif running and not mine and not installed:
        note = "Another download is running. Wait for it or cancel it first."
    elif in_use:
        note = "This model is in use, so it can't be deleted."
    elif selected:
        note = "This model is chosen. Choose another one first to delete it."
    return Actions(download=can_download, cancel=mine, use=installed and not selected and (not in_use or other_chosen),
                   delete=(installed or st.state == "partial") and not in_use and not selected and not mine, note=note)
