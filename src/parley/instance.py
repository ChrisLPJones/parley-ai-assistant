"""Single running tray instance per Parley home, via a named mutex (Windows).

The name is `Local\\Parley-<8 hex of sha256(home)>`, so a throwaway PARLEY_HOME never collides with the real app,
except the DEFAULT home (no PARLEY_HOME): it has the fixed name `DEFAULT_MUTEX`, because the installer's `AppMutex=`
(packaging/parley.iss) must name it without knowing the installing user's paths. A throwaway home therefore never makes
the installer or uninstaller ask to close anything. The mutex is used only as "exists or not" (the first creator keeps the handle for the life of the process), so there is no
owner or abandoned-mutex case to handle. A second launch finds the first one's window and posts it a registered
message ("open settings"); neither the mutex nor the message carries any data.
"""
import hashlib
import logging
import os
import sys
import time
from pathlib import Path

log = logging.getLogger(__name__)

DEFAULT_MUTEX = "Local\\Parley-default"  # keep in sync with AppMutex in packaging/parley.iss (a test checks it)
RESTART_WAIT_S = 20.0  # `--restarted`: the old process is still tearing down
_ERROR_ALREADY_EXISTS = 183


def mutex_name(home: Path) -> str:
    from . import config  # imported here, not at module level, to keep this module's import light

    key = os.path.normcase(os.path.abspath(str(home)))
    if key == os.path.normcase(os.path.abspath(str(config.default_app_dir()))):
        return DEFAULT_MUTEX
    return "Local\\Parley-" + hashlib.sha256(key.encode("utf-8", "replace")).hexdigest()[:8]


def _try_create(name: str):
    """A handle if this call created the mutex, None if it already existed (or on a non-Windows system, a dummy)."""
    if sys.platform != "win32":
        return object()
    import win32api
    import win32event

    handle = win32event.CreateMutex(None, False, name)
    if win32api.GetLastError() == _ERROR_ALREADY_EXISTS:
        win32api.CloseHandle(handle)
        return None
    return handle


def acquire(home: Path, wait_s: float = 0.0, *, poll: float = 0.1):
    """Take the instance lock for `home`. Returns a handle to keep alive for the process, or None when another
    instance holds it (after waiting up to `wait_s` seconds for it to go away)."""
    name = mutex_name(home)
    deadline = time.monotonic() + wait_s
    while True:
        handle = _try_create(name)
        if handle is not None:
            return handle
        if time.monotonic() >= deadline:
            return None
        time.sleep(poll)


def wake_existing(retry_s: float = 2.0) -> bool:
    """Ask the running instance to open Settings. The first instance may still be creating its window, so retry briefly."""
    from . import tray

    deadline = time.monotonic() + retry_s
    while True:
        if tray.post_open_settings():
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.1)
