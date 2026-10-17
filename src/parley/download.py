"""Resumable file downloads with a simple progress line."""
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
