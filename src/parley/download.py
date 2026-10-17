"""Resumable file downloads with a simple progress line."""
import hashlib
import logging
import re
import stat
import sys
import tarfile
import threading
import time
import zipfile
from collections.abc import Callable
from pathlib import Path

import httpx

log = logging.getLogger(__name__)


class DownloadCancelled(Exception):
    """The caller set the `cancel` event; the `.part` file is kept so a later call resumes."""


PROGRESS_INTERVAL = 0.5  # seconds between progress reports
Observer = Callable[[str, int, int], None]
_observer: Observer | None = None


def set_observer(fn: Observer | None) -> None:
    """Receive (file name, bytes done, total bytes or 0) from every `download()` that has no `progress` of its own.
    Used by the tray to show first-run model downloads; `total == done` marks the end of a file."""
    global _observer
    _observer = fn


def _console_progress() -> bool:
    err = sys.stderr  # None in a windowed process
    return err is not None and hasattr(err, "isatty") and err.isatty()


def download(url: str, dest: Path, progress: Callable[[int, int], None] | None = None,
             cancel: threading.Event | None = None) -> Path:
    """Resumable download. `progress(done, total)` (total 0 = unknown) is called about twice a second and once at the
    end; `cancel` (an Event) aborts with DownloadCancelled between chunks, keeping the `.part` file."""
    if dest.exists():
        return dest
    obs = _observer
    report = progress or (lambda done, total: obs(dest.name, done, total) if obs else None)
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    have = part.stat().st_size if part.exists() else 0
    headers = {"Range": f"bytes={have}-"} if have else {}
    with httpx.stream("GET", url, headers=headers, follow_redirects=True, timeout=httpx.Timeout(30, read=120)) as r:
        if r.status_code == 416:  # already complete
            part.replace(dest)
            return dest
        r.raise_for_status()
        if r.status_code != 206:
            have = 0  # server ignored the range; start over
        total = have + int(r.headers.get("content-length", 0))
        done, last = have, float('-inf')
        tty = _console_progress()
        with open(part, "ab" if have else "wb") as f:
            for chunk in r.iter_bytes(1 << 20):
                if cancel is not None and cancel.is_set():
                    raise DownloadCancelled(dest.name)
                f.write(chunk)
                done += len(chunk)
                if time.monotonic() - last >= PROGRESS_INTERVAL:
                    last = time.monotonic()
                    _safe(report, done, total)
                    if tty:
                        pct = f"{done / total:5.1%}" if total else ""
                        print(f"\r  {dest.name}: {done / 1e9:.2f} GB {pct}", end="", file=sys.stderr, flush=True)
    if tty:
        print(file=sys.stderr)
    part.replace(dest)
    _safe(report, done, done)
    return dest


def _safe(report: Callable[[int, int], None], done: int, total: int) -> None:
    try:
        report(done, total)
    except Exception:  # a broken progress display must never fail a download
        log.debug("progress callback failed", exc_info=True)


class DownloadError(Exception):
    """A verified download failed (wrong size or hash, or too much data)."""


def download_verified(open_stream, url: str, dest: Path, sha256: str, size: int,
                      progress: Callable[[int, int], None] | None = None,
                      cancel: threading.Event | None = None) -> Path:
    """Stream `url` to `dest`, accepting it only if it is exactly `size` bytes with the given sha256.

    `open_stream(url)` is a context manager yielding a response (lets the caller control TLS and
    redirects). Always starts fresh (no resume), aborts as soon as more than `size` bytes arrive,
    and only moves the file into place after both checks pass; a bad download is deleted.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    digest, done = hashlib.sha256(), 0
    try:
        with open_stream(url) as r, open(part, "wb") as f:
            for chunk in r.iter_bytes(1 << 20):
                if cancel is not None and cancel.is_set():
                    raise DownloadCancelled(dest.name)
                done += len(chunk)
                if progress is not None:
                    _safe(progress, done, size)
                if done > size:
                    raise DownloadError("download is larger than announced")
                digest.update(chunk)
                f.write(chunk)
        if done != size:
            raise DownloadError("download is truncated")
        if digest.hexdigest() != sha256:
            raise DownloadError("download does not match its sha256")
        part.replace(dest)
    except BaseException:
        part.unlink(missing_ok=True)
        raise
    return dest


# ---------- safe archive extraction (llama.cpp runtime) ----------

class ArchiveError(Exception):
    """An archive member is not acceptable (unsafe name, link, too big) or the archive is too large."""


_PART = re.compile(r"[A-Za-z0-9._+-]{1,128}", re.ASCII)
_RESERVED = {"con", "prn", "aux", "nul", "conin$", "conout$", *(f"com{i}" for i in range(1, 10)), *(f"lpt{i}" for i in range(1, 10))}
MAX_MEMBERS = 2000
MAX_MEMBER_BYTES = 2 * 1024**3
MAX_TOTAL_BYTES = 6 * 1024**3
MAX_DEPTH = 6


def _check_name(name: str) -> tuple[str, ...]:
    """Conservative rule for an archive member name: relative, '/'-separated, every part plain ASCII
    (letters, digits, . _ + -), no '..', no drive letters or colons, no Windows device names. Returns the parts."""
    if not isinstance(name, str) or not name or len(name) > 300:
        raise ArchiveError("bad member name")
    parts = tuple(name[:-1].split("/") if name.endswith("/") else name.split("/"))
    if len(parts) > MAX_DEPTH:
        raise ArchiveError(f"member path too deep: {name[:60]!r}")
    for part in parts:
        if (not _PART.fullmatch(part) or part in (".", "..") or part.endswith(".")
                or part.split(".")[0].lower() in _RESERVED):
            raise ArchiveError(f"unsafe member name: {name[:60]!r}")
    return parts


def safe_extract_zip(archive: Path, target: Path) -> int:
    """Extract every member of a zip into `target`, but only if all member names pass `_check_name`, none is a
    symlink or special file or encrypted, and size caps hold (declared and actual). No member is written before the
    whole listing has been validated. Returns the bytes written."""
    target = target.resolve()
    with zipfile.ZipFile(archive) as zf:
        infos = zf.infolist()
        if len(infos) > MAX_MEMBERS:
            raise ArchiveError("too many members")
        seen: set[tuple[str, ...]] = set()
        total = 0
        for info in infos:
            parts = _check_name(info.filename)
            key = tuple(p.lower() for p in parts)
            if key in seen:
                raise ArchiveError(f"duplicate member: {info.filename[:60]!r}")
            seen.add(key)
            mode = info.external_attr >> 16
            kind = stat.S_IFMT(mode)  # 0 when the zip was written without Unix attributes (Windows tools)
            if info.flag_bits & 0x1 or (kind and kind not in (stat.S_IFREG, stat.S_IFDIR)):
                raise ArchiveError(f"member is not a plain file: {info.filename[:60]!r}")
            if info.file_size > MAX_MEMBER_BYTES:
                raise ArchiveError("member too large")
            total += info.file_size
        if total > MAX_TOTAL_BYTES:
            raise ArchiveError("archive too large")
        written = 0
        for info in infos:
            dest = target.joinpath(*_check_name(info.filename))
            if target not in dest.resolve().parents and dest.resolve() != target:
                raise ArchiveError("member escapes the target folder")
            if info.is_dir():
                dest.mkdir(parents=True, exist_ok=True)
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            n = 0
            with zf.open(info) as src, open(dest, "wb") as out:
                while chunk := src.read(1 << 20):
                    n += len(chunk)
                    if n > info.file_size:
                        raise ArchiveError("member is larger than it declares")
                    out.write(chunk)
            if n != info.file_size:
                raise ArchiveError("member is truncated")
            written += n
    return written


def safe_extract_tar(archive: Path, target: Path) -> int:
    """Same rules for a tar(.gz): plain names, regular files, directories and symlinks whose target is a single plain
    file name next to the link; hard links, devices and fifos are refused. Extracted with filter="data" as well."""
    target.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as t:
        members = t.getmembers()
        if len(members) > MAX_MEMBERS:
            raise ArchiveError("too many members")
        seen: set[tuple[str, ...]] = set()
        total = 0
        for m in members:
            parts = _check_name(m.name)
            if tuple(p.lower() for p in parts) in seen:
                raise ArchiveError(f"duplicate member: {m.name[:60]!r}")
            seen.add(tuple(p.lower() for p in parts))
            if m.issym():
                if not _PART.fullmatch(m.linkname) or m.linkname in (".", ".."):
                    raise ArchiveError(f"unsafe link target: {m.name[:60]!r}")
            elif not (m.isreg() or m.isdir()):
                raise ArchiveError(f"member is not a plain file: {m.name[:60]!r}")
            if m.size > MAX_MEMBER_BYTES:
                raise ArchiveError("member too large")
            total += m.size
        if total > MAX_TOTAL_BYTES:
            raise ArchiveError("archive too large")
        t.extractall(target, members=members, filter="data")
    return total
