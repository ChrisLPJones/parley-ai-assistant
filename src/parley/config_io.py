"""Writes changed settings back to config.toml without disturbing the rest of the file.

Read-modify-write with tomlkit: the file is re-read at save time and only the keys the caller changed are
touched, so comments and edits made outside Parley survive. Values are always typed tomlkit items (never
string-built TOML), the write is atomic (temp file + os.replace), and a file that does not parse is never
overwritten. One config.toml.bak keeps the previous version, except while a plaintext llm.api_key is in the file.
"""
import logging
import os
import re
import tempfile
from pathlib import Path

import tomlkit
from tomlkit.exceptions import TOMLKitError

log = logging.getLogger(__name__)


class ConfigFileError(Exception):
    """The config file can't be read or written; the message is a short sentence safe to show."""


def _parse(text: str, path: Path):
    try:
        return tomlkit.parse(text)
    except TOMLKitError as e:
        line = getattr(e, "line", None)
        where = f" at line {line}" if line else ""
        raise ConfigFileError(f"{path.name} has a mistake{where}, so Parley won't overwrite it. Fix or delete the file, then save again.") from None


def _has_plaintext_key(doc) -> bool:
    llm = doc.get("llm")
    key = llm.get("api_key") if isinstance(llm, dict) else None
    return isinstance(key, str) and key != ""


def _typed(value):
    """A tomlkit item for a plain bool / int / float / str; anything else is a programming error."""
    if isinstance(value, (bool, int, float, str)):
        return tomlkit.item(value)
    raise TypeError(f"can't store a {type(value).__name__} in config.toml")


def _split(dotted: str) -> tuple[str, str]:
    table, _, key = dotted.partition(".")
    if not re.fullmatch(r"[a-z_]+", table) or not re.fullmatch(r"[a-z_]+", key):
        raise ValueError(f"bad setting name {dotted!r}")
    return table, key


def save_changes(path: Path, changes: dict[str, object]) -> None:
    """Apply {"table.key": value} to the file at `path` (created if absent). A value of None removes the key
    (the setting goes back to its default). Raises ConfigFileError if the existing file doesn't parse."""
    path = Path(path)
    parsed = {}
    for dotted, value in changes.items():
        parsed[_split(dotted)] = None if value is None else _typed(value)

    old_bytes = path.read_bytes() if path.exists() else None
    try:
        old_text = old_bytes.decode("utf-8") if old_bytes is not None else ""
    except UnicodeDecodeError:
        raise ConfigFileError(f"{path.name} is not valid UTF-8, so Parley won't overwrite it.") from None
    doc = _parse(old_text, path)
    key_before = _has_plaintext_key(doc)

    for (table, key), item in parsed.items():
        section = doc.get(table)
        if section is None:
            if item is None:
                continue
            section = tomlkit.table()
            doc[table] = section
        elif not isinstance(section, dict):
            raise ConfigFileError(f"{path.name}: [{table}] is not a table, so Parley won't overwrite it.")
        if item is None:
            section.pop(key, None)
        else:
            section[key] = item

    out = tomlkit.dumps(doc).encode("utf-8")
    tomlkit.parse(out.decode("utf-8"))  # never write something we could not read back

    path.parent.mkdir(parents=True, exist_ok=True)
    bak = path.with_name(path.name + ".bak")
    if old_bytes is not None and not key_before and not _has_plaintext_key(doc):
        _atomic_write(bak, old_bytes)
    _atomic_write(path, out)
    log.info("Saved %d setting(s) to %s", len(parsed), path)


def _atomic_write(path: Path, data: bytes) -> None:
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
