"""Strict SemVer (MAJOR.MINOR.PATCH) parsing and comparison for the update check."""
import re

_SEMVER = re.compile(r"(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)", re.ASCII)
_MAX_PART = 10**9  # keeps comparisons and hostile inputs sane


def parse_version(text: object) -> tuple[int, int, int]:
    """Return (major, minor, patch). Raises ValueError for anything but plain MAJOR.MINOR.PATCH."""
    if not isinstance(text, str) or len(text) > 32 or not _SEMVER.fullmatch(text):
        raise ValueError("not a MAJOR.MINOR.PATCH version")
    major, minor, patch = (int(x) for x in text.split("."))
    if max(major, minor, patch) >= _MAX_PART:
        raise ValueError("version number too large")
    return major, minor, patch


def compare_versions(a: str, b: str) -> int:
    """-1, 0 or 1 as `a` is older than, equal to or newer than `b`."""
    pa, pb = parse_version(a), parse_version(b)
    return (pa > pb) - (pa < pb)
