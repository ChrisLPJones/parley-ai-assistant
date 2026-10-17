"""Install a new app version next to the running one and switch to it atomically. Not wired into the app yet.

Steps (security notes in agent-reports/architecture-installer-updates.md, "Self-replacement and rollback"):
  1. build `app/<ver>.partial`: hardlink unchanged files from the current version (re-hashing the source right
     before linking; copy if links fail) and download only the changed blobs (hash and size verified);
  2. verify the whole tree against the index (no extras, no links), write the index copy used by the launcher;
  3. smoke-run `--self-check` (injectable), then verify the tree again (code ran from it in the meantime);
  4. rename to `app/<ver>` and flip `current.json` (state `pending`, `previous` = old) with os.replace;
  5. only after the flip: `Updater.note_verified`.
The running version is never modified. Any failure before the flip deletes the partial dir and leaves
`current.json` and the old app tree byte-identical. The app calls `confirm()` once it reached "Ready";
the launcher (launcher.py) rolls back a pending version that crashes or never confirms.
Callers must catch `UPDATE_ERRORS`.
"""
import hashlib
import json
import logging
import os
import shutil
import stat
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path

import httpx

from . import launcher
from .download import DownloadError
from .launcher import INDEX_NAME, Pointer, app_path, read_pointer, write_pointer
from .update import IndexFile, Manifest, UpdateError, Updater, check_reparse_free
from .version import compare_versions

log = logging.getLogger(__name__)

UPDATE_ERRORS = (UpdateError, DownloadError, httpx.HTTPError, OSError)
SELF_CHECK_TIMEOUT = 30.0


class InstallError(UpdateError):
    """The new version could not be built or did not pass its checks (the old version is untouched)."""


# ---------- helpers ----------

def _rmtree(path: Path) -> None:
    def handler(func, p, _exc):  # read-only files (common in app trees) block deletion on Windows
        try:
            os.chmod(p, stat.S_IWRITE)
            func(p)
        except OSError:
            pass
    if sys.version_info >= (3, 12):
        shutil.rmtree(path, onexc=handler)
    else:
        shutil.rmtree(path, onerror=handler)


def harden_dir(path: Path) -> None:
    """Make `path` user-only: chmod 700, or on Windows drop inherited ACEs and grant only this user, SYSTEM and
    Administrators. Raises InstallError if that fails (we do not install code into a shared folder)."""
    path.mkdir(parents=True, exist_ok=True)
    if sys.platform != "win32":
        os.chmod(path, 0o700)
        return
    user = os.environ.get("USERNAME")
    if not user:
        raise InstallError("cannot tell which user to restrict the install folder to")
    domain = os.environ.get("USERDOMAIN")
    who = f"{domain}\\{user}" if domain else user
    try:
        r = subprocess.run(["icacls", str(path), "/inheritance:r", "/grant:r", f"{who}:(OI)(CI)F",
                            "/grant:r", "*S-1-5-18:(OI)(CI)F", "/grant:r", "*S-1-5-32-544:(OI)(CI)F"],
                           capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError) as e:  # includes TimeoutExpired
        raise InstallError("could not restrict the install folder to this user") from e
    if r.returncode != 0:
        raise InstallError("could not restrict the install folder to this user")


def _replace_dir(src: Path, dst: Path, tries: int = 5, delay: float = 0.2) -> None:
    """os.replace with a short retry: antivirus scanners briefly hold handles on freshly written trees. Fails closed."""
    for i in range(tries):
        try:
            os.replace(src, dst)
            return
        except OSError:
            if i == tries - 1:
                raise
            time.sleep(delay)


def run_self_check(appdir: Path) -> bool:
    """Default smoke run: `<appdir>/parley-app.exe --self-check` must exit 0 within 30 s."""
    try:
        r = subprocess.run([str(appdir / launcher.CHILD_EXE), "--self-check"], cwd=appdir, capture_output=True,
                           timeout=SELF_CHECK_TIMEOUT)
    except (OSError, subprocess.SubprocessError):
        return False
    return r.returncode == 0


def _index_bytes(version: str, files: list[IndexFile]) -> bytes:
    return json.dumps({"schema": 1, "version": version,
                       "files": [{"path": f.path, "sha256": f.sha256, "size": f.size} for f in files]},
                      sort_keys=True).encode()


def _entries(files: list[IndexFile]) -> list[tuple[str, str, int]]:
    return [(f.path, f.sha256, f.size) for f in files]


def _put_file(src: Path | None, f: IndexFile, dest: Path, updater: Updater) -> None:
    """Hardlink `src` (re-hashed first) to `dest` when it is unchanged, else download the blob."""
    if src is not None:
        try:
            st = src.lstat()
            ok = stat.S_ISREG(st.st_mode) and st.st_size == f.size and launcher.sha256_file(src) == f.sha256
        except OSError:
            ok = False
        if ok:
            try:
                os.link(src, dest)
                # one inode now serves both versions: read-only, so new code can't edit the old version's file by accident
                os.chmod(dest, stat.S_IREAD)
            except OSError:
                dest.unlink(missing_ok=True)
                shutil.copyfile(src, dest)  # no hardlinks (other file system, policy): copy instead
            if launcher.sha256_file(dest) != f.sha256:  # also covers a source swapped after the first hash
                dest.unlink(missing_ok=True)
                raise InstallError("a reused file changed while it was being copied")
            return
    updater.download_blob(f, dest)


# ---------- public API ----------

def is_skipped(root: Path, version: str) -> bool:
    ptr = read_pointer(root / launcher.POINTER)
    return bool(ptr and version in ptr.skip)


def record_declined(root: Path, version: str) -> None:
    ptr = read_pointer(root / launcher.POINTER)
    if ptr and version not in ptr.declined:
        ptr.declined = launcher._add(ptr.declined, version)
        write_pointer(root, ptr, backup=False)


def install_version(root: Path, updater: Updater, manifest: Manifest, files: list[IndexFile], *,
                    self_check: Callable[[Path], bool] = run_self_check,
                    harden: Callable[[Path], None] = harden_dir) -> Path:
    """Build, verify and switch to `manifest.version`. Returns its app dir. Raises UPDATE_ERRORS members on failure,
    after cleaning up; nothing the old version needs has been touched in that case."""
    version = manifest.version
    cur = read_pointer(root / launcher.POINTER)
    if cur is not None and compare_versions(version, cur.version) <= 0:
        raise InstallError("not newer than the installed version")
    if cur is not None and version in cur.skip:
        raise InstallError("this version was rolled back before")
    harden(root)
    apps = root / launcher.APP_DIR
    harden(apps)
    final, partial = app_path(root, version), apps / f"{version}.partial"
    if final.exists():
        if cur is not None and version in (cur.version, cur.previous):
            raise InstallError("version already installed")
        _rmtree(final)  # leftover of an interrupted switch that no pointer refers to
    if partial.exists():
        _rmtree(partial)
    partial.mkdir()
    flipped = False
    try:
        old = app_path(root, cur.version) if cur is not None else None
        for f in files:
            dest = check_reparse_free(partial, f.path)
            dest.parent.mkdir(parents=True, exist_ok=True)
            src = None
            if old is not None:
                try:
                    src = check_reparse_free(old, f.path)
                except UpdateError:
                    src = None
            _put_file(src, f, dest, updater)
        index = _index_bytes(version, files)
        (partial / INDEX_NAME).write_bytes(index)
        entries = _entries(files)
        _verify(partial, entries)
        if not self_check(partial):
            raise InstallError("the new version failed its self-check")
        _verify(partial, entries)  # the self-check ran code from the tree; make sure it is still exactly the index
        _replace_dir(partial, final)
        try:
            ptr = Pointer(version, cur.version if cur else None, "pending", 0,
                          cur.skip if cur else [], cur.declined if cur else [], hashlib.sha256(index).hexdigest())
            write_pointer(root, ptr)  # the atomic flip
            flipped = True
        finally:
            if not flipped:
                _rmtree(final)
    except BaseException:
        if not flipped:
            if partial.exists():
                _rmtree(partial)
        raise
    try:
        updater.note_verified(version)
    except (OSError, ValueError) as e:
        log.warning("Could not record the verified version: %s", e)
    return final


def _verify(appdir: Path, entries: list[tuple[str, str, int]]) -> None:
    try:
        launcher.verify_tree(appdir, entries)
    except launcher.TreeError as e:
        raise InstallError(f"verification failed: {e}") from e


def confirm(root: Path, version: str) -> bool:
    """The running app reached "Ready": mark its version good, then prune. False if it is not a pending current."""
    ptr = read_pointer(root / launcher.POINTER)
    if ptr is None or ptr.version != version or ptr.state != "pending":
        return False
    ptr.state, ptr.attempts = "good", 0
    write_pointer(root, ptr, backup=False)
    prune(root)
    return True


def prune(root: Path) -> None:
    """Keep the current and previous app dirs; delete everything else under app/ (best effort, in-use dirs stay)."""
    ptr = read_pointer(root / launcher.POINTER)
    if ptr is None:
        return
    keep = {ptr.version, ptr.previous}
    try:
        entries = list((root / launcher.APP_DIR).iterdir())
    except OSError:
        return
    for p in entries:
        if p.name not in keep and p.is_dir() and not p.is_symlink():
            try:
                _rmtree(p)
            except OSError as e:
                log.debug("Could not remove %s: %s", p, e)
