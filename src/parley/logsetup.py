"""Logging plumbing shared by the console loop and tray mode.

Log lines are written from a background thread (QueueListener): if the output stops accepting data (text selected in a
console, Pause pressed), only logging waits, never listening, thinking or speaking. Every handler gets the RedactFilter
from credstore, the last line of defence that scrubs anything shaped like an API key.
"""
import io
import logging
import logging.handlers
import queue
import sys
import threading
from pathlib import Path

from . import credstore

# Libraries like huggingface_hub re-enable httpx request logging, so filter at the handler instead
NOISY = ("httpx", "httpcore", "huggingface_hub", "faster_whisper", "phonemizer", "ddgs", "primp", "anthropic", "openai")
LOG_BYTES, LOG_BACKUPS = 1 << 20, 3  # 1 MB x 3 rotated files


def harden(handlers) -> None:
    for handler in handlers:
        handler.addFilter(lambda r: r.levelno >= logging.ERROR or not r.name.startswith(NOISY))
        handler.addFilter(credstore.RedactFilter())


def start_queue(handlers) -> logging.handlers.QueueListener:
    """Route the root logger through a queue to `handlers` on a background thread. Returns the (started) listener."""
    root = logging.getLogger()
    listener = logging.handlers.QueueListener(queue.Queue(), *handlers, respect_handler_level=True)
    root.handlers = [logging.handlers.QueueHandler(listener.queue)]
    listener.start()
    return listener


def log_dir(home: Path) -> Path:
    return home / "logs"


def setup_file_logging(home: Path, *, verbose: bool = False, console=None) -> logging.handlers.QueueListener:
    """Tray mode: rotating `<home>/logs/parley.log` (plus `console`, a stream, when there really is one)."""
    d = log_dir(home)
    d.mkdir(parents=True, exist_ok=True)
    fh = logging.handlers.RotatingFileHandler(d / "parley.log", maxBytes=LOG_BYTES, backupCount=LOG_BACKUPS,
                                              encoding="utf-8", delay=True)
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    handlers = [fh]
    if console is not None:
        ch = logging.StreamHandler(console)
        ch.setFormatter(logging.Formatter("%(asctime)s %(message)s", "%H:%M:%S"))
        handlers.append(ch)
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    if verbose:
        logging.getLogger("parley").setLevel(logging.DEBUG)
    harden(handlers)
    return start_queue(handlers)


# ---------- stdout/stderr that may not exist ----------

class LogStream(io.TextIOBase):
    """Stand-in for sys.stdout/stderr when the process has no console (None): lines go to the log.
    isatty() is False, so download progress, tqdm and argparse work. Redaction happens in the log handlers."""

    encoding = "utf-8"
    errors = "replace"

    def __init__(self, name: str, level: int) -> None:
        self._logger, self._level, self._buf = logging.getLogger(name), level, ""
        self._busy = threading.local()

    def writable(self) -> bool:
        return True

    def isatty(self) -> bool:
        return False

    def write(self, s) -> int:
        if not isinstance(s, str):
            s = str(s)
        if getattr(self._busy, "on", False):  # a logging handler writing to us again: drop, never loop
            return len(s)
        self._busy.on = True
        try:
            self._buf += s
            *lines, self._buf = self._buf.split("\n")
            for line in lines:
                line = line.rstrip("\r")
                if line.strip():
                    self._logger.log(self._level, "%s", line[:2000])
            if len(self._buf) > 4000:
                self._logger.log(self._level, "%s", self._buf)
                self._buf = ""
        finally:
            self._busy.on = False
        return len(s)

    def flush(self) -> None:
        if self._buf.strip() and not getattr(self._busy, "on", False):
            self._busy.on = True
            try:
                self._logger.log(self._level, "%s", self._buf.rstrip())
            finally:
                self._busy.on = False
        self._buf = ""


def install_stdio_shim() -> bool:
    """Replace a missing sys.stdout/sys.stderr (windowed process) with log-writing streams. True if anything changed."""
    changed = False
    if sys.stdout is None:
        sys.stdout, changed = LogStream("parley.stdout", logging.INFO), True
    if sys.stderr is None:
        sys.stderr, changed = LogStream("parley.stderr", logging.WARNING), True
    return changed
