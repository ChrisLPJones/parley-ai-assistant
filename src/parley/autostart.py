"""Start Parley with Windows: a per-user `Run` registry value (HKCU, no admin) that points at the installed launcher.

Value `Parley` under `Software\\Microsoft\\Windows\\CurrentVersion\\Run` = the quoted path of the install-root `Parley.exe`
(the launcher, so updates never break it). Packaged copies only: a source checkout has no launcher, so `status()` says
unavailable and the settings window disables the control. If the user switched the entry off in Task Manager > Startup
apps, Windows records that under `Explorer\\StartupApproved\\Run` (first byte odd = disabled; **unverified** on every Windows
build); that counts as disabled here, and turning it on again clears the mark. The value name and both key paths are
parameters so tests use their own private key and never touch the real `Parley` value.
"""
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

VALUE_NAME = "Parley"
RUN_KEY = r"Software\Microsoft\Windows\CurrentVersion\Run"
APPROVED_KEY = r"Software\Microsoft\Windows\CurrentVersion\Explorer\StartupApproved\Run"
LAUNCHER = "Parley.exe"


@dataclass(frozen=True)
class State:
    available: bool  # can it be set here at all (Windows, packaged copy)
    enabled: bool  # the value exists, points at the launcher and the user has not disabled it in Task Manager
    reason: str = ""  # why it is unavailable, or a note (e.g. disabled in Task Manager)


def launcher_path() -> Path | None:
    """`<install root>/Parley.exe` of a packaged copy, else None."""
    from . import autoupdate

    root = autoupdate.install_root()
    exe = root / LAUNCHER if root else None
    return exe if exe is not None and exe.is_file() else None


def command_for(exe: Path) -> str:
    text = str(exe)
    if '"' in text or "\0" in text:
        raise ValueError("The install path contains characters that can't be used for a startup entry.")
    return f'"{text}"'


def _winreg():
    if sys.platform != "win32":
        return None
    import winreg

    return winreg


def _read(winreg, key_path: str, name: str):
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, key_path) as k:
            return winreg.QueryValueEx(k, name)[0]
    except OSError:
        return None


def status(name: str = VALUE_NAME, exe: Path | None = None, *, run_key: str = RUN_KEY, approved_key: str = APPROVED_KEY,
           require_launcher: bool = True) -> State:
    """Current state. `exe` is the launcher expected in the value (default: this install's)."""
    winreg = _winreg()
    if winreg is None:
        return State(False, False, "Starting with Windows is only available on Windows.")
    exe = exe or (launcher_path() if require_launcher else None)
    if exe is None and require_launcher:
        return State(False, False, "Only available in the installed copy of Parley (this one runs from source).")
    value = _read(winreg, run_key, name)
    if not isinstance(value, str) or not value.strip():
        return State(True, False)
    if exe is not None and value.strip() != command_for(exe):
        return State(True, False, "Another program is set to start with that name; turning this on replaces it.")
    approved = _read(winreg, approved_key, name)
    if isinstance(approved, (bytes, bytearray)) and approved and approved[0] % 2 == 1:
        return State(True, False, "Turned off in Windows (Task Manager > Startup apps); turning this on re-enables it.")
    return State(True, True)


def set_enabled(on: bool, name: str = VALUE_NAME, exe: Path | None = None, *, run_key: str = RUN_KEY,
                approved_key: str = APPROVED_KEY, require_launcher: bool = True) -> None:
    """Add or remove the Run value. Raises OSError / ValueError with a message safe to show."""
    winreg = _winreg()
    if winreg is None:
        raise OSError("Starting with Windows is only available on Windows.")
    if on:
        exe = exe or (launcher_path() if require_launcher else None)
        if exe is None:
            raise OSError("Only available in the installed copy of Parley.")
        with winreg.CreateKeyEx(winreg.HKEY_CURRENT_USER, run_key, 0, winreg.KEY_SET_VALUE) as k:
            winreg.SetValueEx(k, name, 0, winreg.REG_SZ, command_for(exe))
        _delete(winreg, approved_key, name)  # the user just asked for it: clear an old "disabled in Task Manager"
        log.info("Start with Windows: on")
    else:
        _delete(winreg, run_key, name)
        log.info("Start with Windows: off")


def _delete(winreg, key_path: str, name: str) -> None:
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, key_path, 0, winreg.KEY_SET_VALUE) as k:
            winreg.DeleteValue(k, name)
    except FileNotFoundError:
        pass
