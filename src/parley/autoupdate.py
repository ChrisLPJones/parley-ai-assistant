"""Wires the update client and installer into the running app (built copies only).

After Parley reports Ready: confirm a freshly installed version, tell the user about a rollback, and (once per
launch, in a background thread) check for a newer version. A native Yes/No box asks before anything is installed;
nothing is shown or installed while Parley is busy answering. The feed URL comes only from `_build.py`
(`update.build_info()`), never from config. A git checkout has no `_build.py` and no install root: no flow at all.
Security notes: agent-reports/architecture-installer-updates.md ("Check/prompt flow", "Threat model").
"""
import json
import logging
import subprocess
import sys
import threading
from collections.abc import Callable
from pathlib import Path

from . import __version__, install, launcher, update
from .config import Config
from .install import UPDATE_ERRORS
from .update import Manifest
from .version import compare_versions

log = logging.getLogger(__name__)

STATE_FILE = "update_state.json"
NOTICE_FILE = "rollback_notified.json"
LAUNCHER_EXE = "Parley.exe"
RESTARTED_FLAG = "--restarted"
MAX_PROMPT_NOTES = 1500
_MB_YESNO, _MB_ICONQUESTION, _MB_SETFOREGROUND, _MB_TOPMOST, _IDYES = 0x4, 0x20, 0x10000, 0x40000, 6


# ---------- environment ----------

def install_root() -> Path | None:
    """The install folder of a packaged copy (`<root>/app/<ver>/parley-app.exe`), else None (git checkout)."""
    if not getattr(sys, "frozen", False):
        return None
    root = Path(sys.executable).resolve().parent.parent.parent
    return root if launcher.read_pointer(root / launcher.POINTER) else None


def create(app: "AppLike", cfg: Config) -> "UpdateFlow | None":
    """The flow for this process, or None in a git checkout / unpackaged run (behaves exactly as without updates)."""
    root = install_root()
    if root is None:
        return None
    updater = None
    if cfg.updates.check:
        try:
            updater = update.updater_from_build(root / STATE_FILE)
        except (update.UpdateError, ValueError) as e:  # bad baked-in feed URL: only the check is off, confirm/rollback notice stay
            log.warning("Update checks are off: %s", e)
    return UpdateFlow(app, root, updater)


def message_box_yes_no(title: str, text: str) -> bool:
    """Native, topmost Yes/No box. Blocks only the calling (update) thread. Anything but Yes (or no Windows) is No."""
    if sys.platform != "win32":
        return False
    import ctypes
    flags = _MB_YESNO | _MB_ICONQUESTION | _MB_SETFOREGROUND | _MB_TOPMOST
    return ctypes.windll.user32.MessageBoxW(None, text, title, flags) == _IDYES


def spawn_launcher(root: Path, args: list[str]) -> None:
    """Start the launcher detached; it runs the newly current version. Failure is logged (next start gets it anyway)."""
    exe = root / LAUNCHER_EXE
    if not exe.is_file():
        log.warning("Cannot restart: %s not found; the new version starts next time you open Parley", exe)
        return
    flags = getattr(subprocess, "DETACHED_PROCESS", 0) | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    try:
        subprocess.Popen([str(exe), *args], cwd=root, close_fds=True, creationflags=flags)
    except OSError as e:
        log.warning("Cannot restart: %s", e)


def prompt_text(m: Manifest, current: str) -> str:
    notes = m.notes.strip()
    if len(notes) > MAX_PROMPT_NOTES:
        notes = notes[:MAX_PROMPT_NOTES].rstrip() + "..."
    body = f"\n\n{notes}" if notes else ""
    return f"Parley {m.version} is available (you have {current}).{body}\n\nInstall it now? Parley will restart."


class AppLike:  # the part of app.Parley the flow uses (documentation for tests and fakes)
    def is_busy(self) -> bool: ...
    def say(self, text: str) -> None: ...


# ---------- the flow ----------

class UpdateFlow:
    def __init__(self, app: AppLike, root: Path, updater: update.Updater | None, *,
                 prompt: Callable[[str, str], bool] = message_box_yes_no,
                 launch: Callable[[Path, list[str]], None] = spawn_launcher,
                 poll: float = 1.0, version: str = __version__) -> None:
        self.app, self.root, self.updater = app, root, updater
        self.prompt, self.launch, self.poll, self.version = prompt, launch, poll, version
        self.restart_requested = threading.Event()  # the main loop leaves, then `restart()` runs
        self._stop = threading.Event()
        self._started = False
        self._lock = threading.Lock()

    # -- lifecycle --
    def start(self) -> None:
        """Call once Parley is Ready: confirm, announce a rollback, then check in the background (once per launch)."""
        with self._lock:
            if self._started:
                return
            self._started = True
        self.confirm_and_notice()
        if self.updater is not None:
            threading.Thread(target=self.run_check, daemon=True, name="update-check").start()

    def stop(self) -> None:
        self._stop.set()

    def restart(self) -> None:
        """Start the launcher (which runs the new version). Called after the app has shut down its loop."""
        args = [a for a in sys.argv[1:] if a != RESTARTED_FLAG]
        self.launch(self.root, [*args, RESTARTED_FLAG])  # the new process waits for this one's instance lock

    def wait_idle(self) -> bool:
        """Block until Parley is not busy. False if the flow was stopped first."""
        while self.app.is_busy():
            if self._stop.wait(self.poll):
                return False
        return not self._stop.is_set()

    # -- Ready: confirm + rollback notice --
    def confirm_and_notice(self) -> None:
        try:
            if install.confirm(self.root, self.version):
                log.info("Version %s is confirmed", self.version)
            self._rollback_notice()
        except Exception:  # never let bookkeeping break the app
            log.exception("Update bookkeeping failed")

    def _rollback_notice(self) -> None:
        ptr = launcher.read_pointer(self.root / launcher.POINTER)
        if ptr is None or ptr.version != self.version or ptr.state != "good":
            return
        notified = self._notified()
        failed = [v for v in ptr.skip if compare_versions(v, self.version) > 0 and v not in notified]
        if not failed:
            return
        newest = max(failed, key=lambda v: tuple(int(x) for x in v.split(".")))
        msg = f"Update to {newest} failed; staying on {self.version}."
        log.warning(msg)
        self.app.say(msg)
        try:
            (self.root / NOTICE_FILE).write_text(json.dumps(sorted({*notified, *failed})), "utf-8")
        except OSError as e:
            log.debug("Could not record the rollback notice: %s", e)

    def _notified(self) -> set[str]:
        try:
            items = json.loads((self.root / NOTICE_FILE).read_text("utf-8"))
            return {v for v in items if isinstance(v, str)}
        except (OSError, ValueError, TypeError):
            return set()

    # -- check, ask, install --
    def run_check(self) -> None:
        try:
            self._check_and_install()
        except UPDATE_ERRORS as e:
            log.debug("Update check or install failed quietly: %s", e)
        except Exception:
            log.exception("Update flow failed")

    def check_now(self, notify: Callable[[str], None]) -> None:
        """Manual check (tray menu), meant for a worker thread: same flow as the launch check, but says the result."""
        if self.updater is None:
            notify("Update checks are turned off.")
            return
        try:
            self._check_and_install(notify)
        except UPDATE_ERRORS as e:
            log.info("Manual update check failed: %s", e)
            notify("Could not check for updates.")
        except Exception:
            log.exception("Update flow failed")
            notify("Could not check for updates.")

    def _check_and_install(self, notify: Callable[[str], None] | None = None) -> None:
        u = self.updater
        if u is None or not self.wait_idle():
            return
        m = u.check()  # quiet None when offline, invalid or not newer
        if m is None or install.is_skipped(self.root, m.version):
            if notify:
                notify(f"Parley {self.version} is up to date.")
            return
        if not self.wait_idle():  # recheck right before prompting
            return
        if not self.prompt("Parley update", prompt_text(m, self.version)):
            log.info("Update to %s declined; asking again next launch", m.version)
            return
        if not self.wait_idle():  # the box can stay open a long time; recheck before applying
            return
        try:
            files = u.fetch_index(m)
            install.install_version(self.root, u, m, files)
        except UPDATE_ERRORS as e:
            log.warning("Update to %s failed: %s", m.version, e)
            self.app.say(f"The update to {m.version} didn't install. Staying on {self.version}.")
            return
        log.info("Version %s installed; restarting", m.version)
        self.app.say(f"Updated to version {m.version}. Restarting.")
        if self.wait_idle():  # never cut a conversation (or this sentence) short
            self.restart_requested.set()
