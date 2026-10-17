"""Resumable file downloads with a simple progress line."""
import hashlib
import logging
import sys
import time
from pathlib import Path

import httpx

log = logging.getLogger(__name__)


def download(url: str, dest: Path) -> Path:
    if dest.exists():
        return dest
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
        done, last = have, 0.0
        with open(part, "ab" if have else "wb") as f:
            for chunk in r.iter_bytes(1 << 20):
                f.write(chunk)
                done += len(chunk)
                if sys.stderr.isatty() and time.monotonic() - last > 0.5:
                    last = time.monotonic()
                    pct = f"{done / total:5.1%}" if total else ""
                    print(f"\r  {dest.name}: {done / 1e9:.2f} GB {pct}", end="", file=sys.stderr, flush=True)
    if sys.stderr.isatty():
        print(file=sys.stderr)
    part.replace(dest)
    return dest


class DownloadError(Exception):
    """A verified download failed (wrong size or hash, or too much data)."""


def download_verified(open_stream, url: str, dest: Path, sha256: str, size: int) -> Path:
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
                done += len(chunk)
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
