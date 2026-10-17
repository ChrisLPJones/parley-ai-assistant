"""Update client core: fetch and validate the version manifest and files index, download verified files.

Not wired into the app yet. Security notes (see agent-reports/architecture-installer-updates.md):
everything from the server is untrusted. The manifest is capped and strictly validated, URLs are built
only from validated relative paths on the feed's own host, https is required, redirects stay on the
same origin, every download is checked against a sha256 and size, and a version that is not newer
than the installed one (or than the highest one already verified) is refused. The `signature` field
is parsed but NOT verified yet: without it, hashes only detect corruption, not a compromised server.
"""
import hashlib
import json
import logging
import re
import unicodedata
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from urllib.parse import urljoin, urlsplit

import httpx

from . import __version__
from .download import download_verified
from .version import compare_versions, parse_version

log = logging.getLogger(__name__)

PLATFORM_KEY = "windows-x64"
MAX_MANIFEST_BYTES = 64 * 1024
MAX_INDEX_BYTES = 8 * 1024 * 1024
MAX_FILES = 20_000
MAX_FILE_BYTES = 2 * 1024**3
MAX_TOTAL_BYTES = 4 * 1024**3
MAX_NOTES = 4000
MAX_REDIRECTS = 3
TIMEOUT = httpx.Timeout(10.0, read=30.0)
_HEX64 = re.compile(r"[0-9a-f]{64}", re.ASCII)
_FILE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", re.ASCII)
_RESERVED = {"con", "prn", "aux", "nul", *(f"com{i}" for i in range(1, 10)), *(f"lpt{i}" for i in range(1, 10))}
_BAD_CHARS = set('<>:"|?*\\')
_LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}


class UpdateError(Exception):
    """The server's answer is invalid or unsafe (never shown to the user in detail)."""


@dataclass(frozen=True)
class PlatformEntry:
    file: str
    sha256: str
    size: int
    index: str | None = None
    index_sha256: str | None = None


@dataclass(frozen=True)
class Manifest:
    version: str
    notes: str
    released: str
    platform: PlatformEntry
    signature: str | None  # reserved; not verified yet


@dataclass(frozen=True)
class IndexFile:
    path: str
    sha256: str
    size: int


@dataclass(frozen=True)
class BuildInfo:
    feed_url: str
    platform: str = PLATFORM_KEY
    public_key: str | None = None  # reserved for signature verification


@dataclass
class UpdateState:
    """Persisted replay protection: the highest version whose files were fully verified."""
    highest_seen: str | None = field(default=None)


# ---------- validation helpers ----------

def _is_int(v: object) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _sha(v: object, what: str) -> str:
    if not isinstance(v, str) or not _HEX64.fullmatch(v):
        raise UpdateError(f"bad {what}")
    return v


def _size(v: object, what: str, limit: int = MAX_FILE_BYTES) -> int:
    if not _is_int(v) or not 0 <= v <= limit:  # type: ignore[operator]
        raise UpdateError(f"bad {what}")
    return v  # type: ignore[return-value]


def _text(v: object, what: str, limit: int) -> str:
    """Plain text only: strip control characters other than newline and tab."""
    if v is None:
        return ""
    if not isinstance(v, str) or len(v) > limit:
        raise UpdateError(f"bad {what}")
    return "".join(c for c in v if c in "\n\t" or unicodedata.category(c)[0] != "C")


def validate_rel_path(path: object) -> str:
    """A safe relative path using forward slashes: no `..`, absolute or drive paths, reserved names."""
    if not isinstance(path, str) or not path or len(path) > 240:
        raise UpdateError("bad path")
    if path.startswith("/") or "//" in path or any(c in _BAD_CHARS or ord(c) < 32 for c in path):
        raise UpdateError("bad path")
    for part in path.split("/"):
        if part in ("", ".", "..") or part != part.rstrip(". ") or part.split(".")[0].lower() in _RESERVED:
            raise UpdateError("bad path")
    return path


def validate_index_files(raw: object) -> list[IndexFile]:
    """Validate the `files` list: bounded, safe paths, no case-insensitive duplicates, no file/dir clashes."""
    if not isinstance(raw, list) or not 0 < len(raw) <= MAX_FILES:
        raise UpdateError("bad file list")
    files: list[IndexFile] = []
    seen: set[str] = set()
    dirs: set[str] = set()
    total = 0
    for item in raw:
        if not isinstance(item, dict):
            raise UpdateError("bad file entry")
        path = validate_rel_path(item.get("path"))
        entry = IndexFile(path, _sha(item.get("sha256"), "file hash"), _size(item.get("size"), "file size"))
        total += entry.size
        key = path.casefold()
        if key in seen or key in dirs:
            raise UpdateError("colliding paths")
        parts = key.split("/")
        for i in range(1, len(parts)):
            prefix = "/".join(parts[:i])
            if prefix in seen:
                raise UpdateError("colliding paths")
            dirs.add(prefix)
        seen.add(key)
        files.append(entry)
    if total > MAX_TOTAL_BYTES:
        raise UpdateError("update too large")
    return files


def parse_index(raw: bytes, expected_sha256: str, version: str) -> list[IndexFile]:
    """Parse files.json after checking its hash (against the manifest) and that it is for `version`."""
    if len(raw) > MAX_INDEX_BYTES:
        raise UpdateError("index too large")
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise UpdateError("index does not match its sha256")
    doc = _load_json(raw)
    if not _is_int(doc.get("schema")) or doc["schema"] != 1 or doc.get("version") != version:
        raise UpdateError("index is for another version")
    return validate_index_files(doc.get("files"))


def _load_json(raw: bytes) -> dict:
    try:
        doc = json.loads(raw.decode("utf-8"))
    except (ValueError, RecursionError) as e:  # UnicodeDecodeError is a ValueError
        raise UpdateError("not valid JSON") from e
    if not isinstance(doc, dict):
        raise UpdateError("not a JSON object")
    return doc


def _platform_entry(src: dict, version: str) -> PlatformEntry:
    name = src.get("file")
    if not isinstance(name, str) or not _FILE_NAME.fullmatch(name) or name.startswith("."):
        raise UpdateError("bad file name")
    index, index_sha = src.get("index"), src.get("index_sha256")
    if index is not None:
        index = validate_rel_path(index)
        index_sha = _sha(index_sha, "index hash")
    elif index_sha is not None:
        raise UpdateError("index hash without index")
    return PlatformEntry(name, _sha(src.get("sha256"), "installer hash"), _size(src.get("size"), "installer size"),
                         index, index_sha)


def parse_manifest(raw: bytes, platform: str = PLATFORM_KEY) -> Manifest:
    """Validate version.json. Unknown keys are ignored; only the chosen platform entry is read."""
    if len(raw) > MAX_MANIFEST_BYTES:
        raise UpdateError("manifest too large")
    doc = _load_json(raw)
    if not _is_int(doc.get("schema")) or doc["schema"] != 1:
        raise UpdateError("unsupported manifest schema")
    version = doc.get("version")
    try:
        parse_version(version)
    except ValueError as e:
        raise UpdateError("bad version") from e
    platforms = doc.get("platforms")
    if platforms is None:
        src = doc if platform == PLATFORM_KEY else None  # legacy: top level is the Windows installer
    elif isinstance(platforms, dict):
        src = platforms.get(platform)
    else:
        raise UpdateError("bad platforms")
    if not isinstance(src, dict):
        raise UpdateError("no build for this platform")
    sig = doc.get("signature")
    if sig is not None and (not isinstance(sig, str) or len(sig) > 1024):
        raise UpdateError("bad signature field")
    return Manifest(version, _text(doc.get("notes"), "notes", MAX_NOTES),  # type: ignore[arg-type]
                    _text(doc.get("released"), "released date", 64), _platform_entry(src, version), sig)


def check_reparse_free(root: Path, rel: str) -> Path:
    """Return root/rel if it stays inside root and no existing component is a symlink or junction."""
    root = root.resolve()
    target = root.joinpath(*PurePosixPath(validate_rel_path(rel)).parts)
    cur = root
    for part in PurePosixPath(rel).parts:
        cur = cur / part
        if cur.is_symlink() or (hasattr(cur, "is_junction") and cur.is_junction()):
            raise UpdateError("path crosses a link")
    if not target.resolve().is_relative_to(root):
        raise UpdateError("path escapes the install folder")
    return target


# ---------- build info ----------

def build_info() -> BuildInfo | None:
    """Read `parley/_build.py` (written only into packaged builds). Absent (a git checkout) = updates off."""
    try:
        from . import _build  # type: ignore[attr-defined]
    except ImportError:
        return None
    url = getattr(_build, "FEED_URL", "")
    if not isinstance(url, str) or not url:
        return None
    return BuildInfo(url, getattr(_build, "PLATFORM", PLATFORM_KEY), getattr(_build, "PUBLIC_KEY", None))


def updater_from_build(state_path: Path) -> "Updater | None":
    info = build_info()
    return Updater(info.feed_url, __version__, info.platform, state_path) if info else None


# ---------- client ----------

class Updater:
    """Checks one feed. `_allow_http_localhost` is a test hook only; never set it from config or env."""

    def __init__(self, feed_url: str, current_version: str, platform: str = PLATFORM_KEY,
                 state_path: Path | None = None, *, _allow_http_localhost: bool = False) -> None:
        parse_version(current_version)
        self.current = current_version
        self.platform = platform
        self.state_path = state_path
        self._http_local = _allow_http_localhost
        self.feed_url = feed_url
        self._origin = self._check_url(feed_url)
        self._base = feed_url.rsplit("/", 1)[0] + "/"

    # -- URL safety --
    def _check_url(self, url: str) -> tuple[str, str, int | None]:
        try:
            u = urlsplit(url)
            host, port = u.hostname, u.port
        except ValueError as e:
            raise UpdateError("bad URL") from e
        if not host or u.username or u.password:
            raise UpdateError("bad URL")
        if u.scheme == "http" and not (self._http_local and host in _LOCAL_HOSTS):
            raise UpdateError("updates need https")
        if u.scheme not in ("http", "https"):
            raise UpdateError("updates need https")
        return u.scheme, host, port

    @contextmanager
    def _open(self, url: str):
        """GET with certificate checks, no auto-redirects; redirects only to the same origin."""
        headers = {"User-Agent": f"parley/{self.current} {self.platform}"}
        with httpx.Client(timeout=TIMEOUT, follow_redirects=False, headers=headers) as client:
            for _ in range(MAX_REDIRECTS + 1):
                if self._check_url(url) != self._origin:
                    raise UpdateError("redirect to another origin")
                r = client.send(client.build_request("GET", url), stream=True)
                if r.is_redirect:
                    loc = r.headers.get("location")
                    r.close()
                    if not loc:
                        raise UpdateError("bad redirect")
                    url = urljoin(url, loc)
                    continue
                try:
                    r.raise_for_status()
                    yield r
                finally:
                    r.close()
                return
            raise UpdateError("too many redirects")

    def _get_bytes(self, url: str, limit: int) -> bytes:
        buf = bytearray()
        with self._open(url) as r:
            declared = r.headers.get("content-length", "")
            if declared.isdigit() and int(declared) > limit:
                raise UpdateError("response too large")
            for chunk in r.iter_bytes(8192):
                buf += chunk
                if len(buf) > limit:
                    raise UpdateError("response too large")
        return bytes(buf)

    # -- state --
    def load_state(self) -> UpdateState:
        try:
            doc = json.loads(self.state_path.read_text("utf-8")) if self.state_path else {}
            seen = doc.get("highest_seen")
            parse_version(seen)
            return UpdateState(seen)
        except (OSError, ValueError, AttributeError, TypeError):
            return UpdateState()

    def floor(self) -> str:
        """Oldest version we would accept: the newer of the installed and the highest verified one."""
        seen = self.load_state().highest_seen
        return seen if seen and compare_versions(seen, self.current) > 0 else self.current

    def note_verified(self, version: str) -> None:
        """Call once a version's files are fully verified; later offers must be newer than it."""
        parse_version(version)
        if self.state_path and compare_versions(version, self.floor()) > 0:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.state_path.with_name(self.state_path.name + ".tmp")
            tmp.write_text(json.dumps({"highest_seen": version}), "utf-8")
            tmp.replace(self.state_path)

    # -- operations --
    def check(self) -> Manifest | None:
        """Newer manifest for this platform, or None (up to date, offline, or the answer was invalid)."""
        try:
            m = parse_manifest(self._get_bytes(self.feed_url, MAX_MANIFEST_BYTES), self.platform)
        except (httpx.HTTPError, OSError) as e:
            log.debug("Update check failed: %s", e)
            return None
        except UpdateError as e:
            log.warning("Ignoring update manifest: %s", e)
            return None
        if compare_versions(m.version, self.floor()) <= 0:
            log.debug("No newer version (feed has %s, floor %s)", m.version, self.floor())
            return None
        return m

    def fetch_index(self, m: Manifest) -> list[IndexFile]:
        """Download and verify the per-version files index. Raises UpdateError / httpx errors."""
        if not m.platform.index or not m.platform.index_sha256:
            raise UpdateError("no files index")
        raw = self._get_bytes(urljoin(self._base, m.platform.index), MAX_INDEX_BYTES)
        return parse_index(raw, m.platform.index_sha256, m.version)

    def download_installer(self, m: Manifest, dest_dir: Path) -> Path:
        e = m.platform
        return download_verified(self._open, urljoin(self._base, e.file), dest_dir / e.file, e.sha256, e.size)

    def download_blob(self, f: IndexFile, dest: Path) -> Path:
        """Fetch the content-addressed blob for one index file into `dest` (hash and size verified)."""
        return download_verified(self._open, urljoin(self._base, f"blobs/{f.sha256}"), dest, f.sha256, f.size)
