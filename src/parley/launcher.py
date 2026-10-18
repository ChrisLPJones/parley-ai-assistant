"""Tiny launcher for installed copies: stdlib only, so it almost never has to change (and is not updated in place).

Layout under the install root (see agent-reports/architecture-installer-updates.md):
    current.json        pointer: {version, previous, state: good|pending, attempts, skip, declined, index_sha256}
    current.json.prev   copy of the last pointer that was valid before the latest change (corruption fallback)
    app/<ver>/          one complete app tree per version; the running version is never modified

A `pending` version (just installed, not yet confirmed by the app reaching "Ready") that exits non-zero, or that
has been started `MAX_ATTEMPTS` times without confirming, is rolled back to `previous` and recorded in `skip`.
The pointer is only ever replaced atomically (temp file + os.replace). Keep this module free of third-party
imports and of imports from the rest of the package: it is shared by `install.py` and the frozen launcher.
"""
import hashlib
import json
import os
import re
import stat
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

POINTER = "current.json"
POINTER_BACKUP = "current.json.prev"
APP_DIR = "app"
INDEX_NAME = ".parley-index.json"
CHILD_EXE = "parley-app.exe"  # console exe: CLI, --ask, --self-check (install smoke test, release smoke)
RUN_EXE = "parley-tray.exe"  # windowed exe the launcher starts: same app tree, implies --tray
MAX_ATTEMPTS = 3
MAX_LIST = 200  # skip / declined are bounded so a corrupt file can't grow without limit
_VERSION = re.compile(r"(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)", re.ASCII)
_HEX64 = re.compile(r"[0-9a-f]{64}", re.ASCII)
_REPARSE = 0x400  # FILE_ATTRIBUTE_REPARSE_POINT (symlinks, junctions)


class LauncherError(Exception):
    """No usable version to start, or the pointer state is unusable."""


class TreeError(LauncherError):
    """An app tree does not match its index."""


@dataclass
class Pointer:
    version: str
    previous: str | None = None
    state: str = "good"
    attempts: int = 0
    skip: list[str] = field(default_factory=list)
    declined: list[str] = field(default_factory=list)
    index_sha256: str | None = None  # sha256 of app/<version>/.parley-index.json, checked before a pending start

    def to_json(self) -> str:
        return json.dumps({"version": self.version, "previous": self.previous, "state": self.state,
                           "attempts": self.attempts, "skip": self.skip, "declined": self.declined,
                           "index_sha256": self.index_sha256}, indent=1)


def _ver(v: object, allow_none: bool = False) -> str | None:
    if v is None and allow_none:
        return None
    if not isinstance(v, str) or len(v) > 32 or not _VERSION.fullmatch(v):
        raise ValueError("bad version")
    return v


def _ver_list(v: object) -> list[str]:
    if v is None:
        return []
    if not isinstance(v, list) or len(v) > MAX_LIST:
        raise ValueError("bad list")
    return [_ver(x) for x in v]  # type: ignore[misc]


def parse_pointer(raw: str) -> Pointer:
    """Strict parse; raises ValueError for anything unexpected."""
    doc = json.loads(raw)
    if not isinstance(doc, dict):
        raise ValueError("not an object")
    state, attempts, idx = doc.get("state", "good"), doc.get("attempts", 0), doc.get("index_sha256")
    if state not in ("good", "pending") or not isinstance(attempts, int) or isinstance(attempts, bool) \
            or not 0 <= attempts <= 1000:
        raise ValueError("bad state")
    if idx is not None and (not isinstance(idx, str) or not _HEX64.fullmatch(idx)):
        raise ValueError("bad index hash")
    return Pointer(_ver(doc.get("version")), _ver(doc.get("previous"), True), state, attempts,  # type: ignore[arg-type]
                   _ver_list(doc.get("skip")), _ver_list(doc.get("declined")), idx)


def read_pointer(path: Path) -> Pointer | None:
    try:
        return parse_pointer(path.read_text("utf-8"))
    except (OSError, ValueError, RecursionError):  # JSONDecodeError and UnicodeDecodeError are ValueErrors
        return None


def write_pointer(root: Path, ptr: Pointer, *, backup: bool = True) -> None:
    """Atomically replace current.json. With `backup`, the pointer being replaced (if valid) becomes current.json.prev."""
    root.mkdir(parents=True, exist_ok=True)
    if backup:
        old = read_pointer(root / POINTER)
        if old is not None:
            _atomic_write(root / POINTER_BACKUP, old.to_json())
    _atomic_write(root / POINTER, ptr.to_json())


def _atomic_write(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def app_path(root: Path, version: str) -> Path:
    return root / APP_DIR / version


def _has_app(root: Path, version: str | None) -> bool:
    return bool(version) and app_path(root, version).is_dir()  # type: ignore[arg-type]


def resolve(root: Path) -> Pointer:
    """The pointer to start from. Falls back (and repairs current.json) when it is corrupt or its app tree is gone:
    first to `previous`, then to the backup pointer, then to the newest app dir. Raises LauncherError if nothing is left."""
    ptr = read_pointer(root / POINTER)
    if ptr is not None and _has_app(root, ptr.version):
        return ptr
    if ptr is not None and ptr.previous and _has_app(root, ptr.previous):
        fixed = Pointer(ptr.previous, None, "good", 0, _add(ptr.skip, ptr.version), ptr.declined)
    else:
        back = read_pointer(root / POINTER_BACKUP)
        if back is not None and _has_app(root, back.version):
            fixed = Pointer(back.version, back.previous if _has_app(root, back.previous) else None, "good", 0,
                            back.skip, back.declined)
        else:
            found = _newest_app(root)
            if found is None:
                raise LauncherError("no installed version found")
            fixed = Pointer(found)
    write_pointer(root, fixed, backup=False)
    return fixed


def _add(items: list[str], v: str) -> list[str]:
    return items if v in items else [*items, v][-MAX_LIST:]


def _newest_app(root: Path) -> str | None:
    try:
        names = [p.name for p in (root / APP_DIR).iterdir() if p.is_dir() and _VERSION.fullmatch(p.name)]
    except OSError:
        return None
    return max(names, key=lambda n: tuple(int(x) for x in n.split(".")), default=None)


def rollback(root: Path, ptr: Pointer) -> Pointer:
    """Point back at `previous` and record the failed version as skipped. The old state is just restored, never edited."""
    if not _has_app(root, ptr.previous):
        raise LauncherError("nothing to roll back to")
    fixed = Pointer(ptr.previous, None, "good", 0, _add(ptr.skip, ptr.version), ptr.declined)  # type: ignore[arg-type]
    write_pointer(root, fixed, backup=False)
    return fixed


# ---------- tree verification (stdlib copy of the update client's checks) ----------

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def _is_link(path: str) -> bool:
    st = os.lstat(path)
    return stat.S_ISLNK(st.st_mode) or bool(getattr(st, "st_file_attributes", 0) & _REPARSE)


def verify_tree(appdir: Path, entries: list[tuple[str, str, int]], *, ignore: tuple[str, ...] = (INDEX_NAME,)) -> None:
    """Check that `appdir` holds exactly `entries` (path, sha256, size): same files, sizes and hashes, no links, no extras."""
    want = {p: (h, n) for p, h, n in entries}
    seen: set[str] = set()
    root = str(appdir)
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        for d in dirnames:
            if _is_link(os.path.join(dirpath, d)):
                raise TreeError("link in app tree")
        for name in filenames:
            full = os.path.join(dirpath, name)
            rel = os.path.relpath(full, root).replace(os.sep, "/")
            if _is_link(full) or not stat.S_ISREG(os.lstat(full).st_mode):
                raise TreeError("link or special file in app tree")
            if rel in ignore and rel not in want:
                continue
            if rel not in want:
                raise TreeError("unexpected file in app tree")
            sha, size = want[rel]
            if os.lstat(full).st_size != size or sha256_file(Path(full)) != sha:
                raise TreeError("file does not match its hash")
            seen.add(rel)
    if seen != set(want):
        raise TreeError("file missing from app tree")


def read_index_entries(appdir: Path, expected_sha256: str) -> list[tuple[str, str, int]]:
    raw = (appdir / INDEX_NAME).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise TreeError("app index changed")
    doc = json.loads(raw.decode("utf-8"))
    out = []
    for f in doc["files"]:
        p = f["path"]
        if not isinstance(p, str) or not p or p.startswith("/") or "\\" in p or ":" in p \
                or any(part in ("", ".", "..") for part in p.split("/")):
            raise TreeError("bad path in app index")
        out.append((p, f["sha256"], f["size"]))
    return out


def tree_ok(root: Path, ptr: Pointer) -> bool:
    """Full re-verification of a pending version just before it is started."""
    if not ptr.index_sha256:
        return False
    try:
        appdir = app_path(root, ptr.version)
        verify_tree(appdir, read_index_entries(appdir, ptr.index_sha256))
    except (OSError, ValueError, KeyError, TypeError, LauncherError):
        return False
    return True


# ---------- running ----------

def default_child(appdir: Path, args: list[str]) -> list[str]:
    return [str(appdir / RUN_EXE), *args]


def run(root: Path, args: list[str] | None = None, *, child=default_child, runner=subprocess.call,
        max_attempts: int = MAX_ATTEMPTS) -> int:
    """Start the current version and return its exit code, rolling a failing pending version back once."""
    args = list(args or [])
    code = 1
    for _ in range(2):  # the second pass starts the version we rolled back to
        ptr = resolve(root)
        appdir = app_path(root, ptr.version)
        if ptr.state == "pending":
            if ptr.attempts >= max_attempts or not tree_ok(root, ptr):
                rollback(root, ptr)
                continue
            ptr.attempts += 1
            write_pointer(root, ptr, backup=False)  # on disk before the start, so a kill still counts as an attempt
        try:
            code = runner(child(appdir, args))
        except OSError:
            code = 127
        now = read_pointer(root / POINTER)
        if now is not None and now.state == "pending" and now.version == ptr.version and code != 0:
            rollback(root, now)
            continue
        return code
    return code


def windowed() -> bool:
    """True in the frozen windowed launcher: it has no console, so sys.stderr is None and errors need a dialog."""
    return bool(getattr(sys, "frozen", False)) and sys.stderr is None


def report_error(message: str) -> None:
    """Tell the user: a MessageBox in the windowed launcher (nobody sees its stdout), else a line on stderr."""
    if windowed() and sys.platform == "win32":
        try:
            import ctypes

            ctypes.windll.user32.MessageBoxW(None, message, "Parley", 0x10)  # MB_OK | MB_ICONERROR
            return
        except Exception:
            pass
    print(message, file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    root = Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else Path.cwd()
    try:
        code = run(root, sys.argv[1:] if argv is None else argv)
    except LauncherError as e:
        report_error(f"Parley cannot start: {e}")
        return 1
    if code == 127:  # the exe could not be started at all (missing or blocked)
        report_error(f"Parley cannot start: {RUN_EXE} could not be launched. Reinstall Parley or check your antivirus.")
    return code


if __name__ == "__main__":
    sys.exit(main())
