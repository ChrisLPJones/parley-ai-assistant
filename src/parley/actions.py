"""Actions shared by the tray menu and the Settings window's Advanced page, so both do exactly the same thing.

Folders: only the two fixed ones (the log folder and the Parley home), never a path that came from a setting or a message.
Each function returns "" on success or one plain sentence on failure, and never raises. `startfile` is injectable for tests.
"""
import logging
import os
from collections.abc import Callable
from pathlib import Path

from . import logsetup

log = logging.getLogger(__name__)

NO_UPDATES = "Update checks are only available in the installed copy of Parley, once it has started."


def _open(folder: Path, startfile: Callable[[str], None] | None) -> str:
    opener = startfile or getattr(os, "startfile", None)
    if opener is None:
        return "Opening folders needs Windows."
    try:
        folder.mkdir(parents=True, exist_ok=True)
        opener(str(folder))
    except OSError as e:
        log.warning("Could not open %s: %s", folder.name, e)
        return "The folder could not be opened."
    return ""


def open_log_folder(home: Path, startfile: Callable[[str], None] | None = None) -> str:
    return _open(logsetup.log_dir(home), startfile)


def open_parley_folder(home: Path, startfile: Callable[[str], None] | None = None) -> str:
    """The Parley home: config.toml, the downloaded models and the logs."""
    return _open(home, startfile)


def check_for_updates(flow, notify: Callable[[str], None]) -> None:
    """The existing update flow's manual check (blocking: worker thread). `flow` is None in a git checkout or while starting."""
    if flow is None:
        notify(NO_UPDATES)
        return
    flow.check_now(notify)
