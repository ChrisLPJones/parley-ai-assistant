"""API keys in the Windows Credential Manager, never in config.toml, logs or error messages.

Generic credentials (CRED_TYPE_GENERIC, CRED_PERSIST_LOCAL_MACHINE: protected by DPAPI for the same user, not roamed).
The target is "Parley/<backend>"; when PARLEY_HOME is set the prefix also carries 8 hex digits of its sha256, so a
throwaway test home never reads or touches the real keys. On other platforms the store reports unavailable: there is
no plaintext fallback. The backend is injectable so tests use a fake.

Also here: resolve_api_key (stored key > environment variable > legacy llm.api_key), the startup migration of a
plaintext llm.api_key out of config.toml, and a logging filter that scrubs anything shaped like a key.
"""
import hashlib
import logging
import os
import re
import sys
import tomllib
from pathlib import Path
from typing import Protocol
from urllib.parse import urlparse

log = logging.getLogger(__name__)

MAX_BLOB = 2560  # CRED_MAX_CREDENTIAL_BLOB_SIZE
MAX_KEY_CHARS = 512
KEY_BACKENDS = ("anthropic", "openai")
ENV_VARS = {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY"}
DEFAULT_HOSTS = {"anthropic": "api.anthropic.com", "openai": "api.openai.com"}
SOURCE_STORE, SOURCE_ENV, SOURCE_CONFIG = "credential store", "environment variable", "config.toml (legacy)"
_ERROR_NOT_FOUND = 1168


class CredentialError(Exception):
    """The credential store failed. The message is a short sentence and never holds a key."""


class Backend(Protocol):
    def read(self, target: str) -> str | None: ...
    def write(self, target: str, secret: str) -> None: ...
    def delete(self, target: str) -> bool: ...


class WinBackend:
    """win32cred; every pywin32 error becomes a CredentialError with only the Windows error code."""

    def __init__(self):
        import win32cred
        self._w = win32cred

    def _fail(self, op: str, e: Exception) -> CredentialError:
        code = e.args[0] if getattr(e, "args", None) and isinstance(e.args[0], int) else "?"
        return CredentialError(f"The Windows credential store failed to {op} (error {code}).")

    def read(self, target):
        try:
            cred = self._w.CredRead(target, self._w.CRED_TYPE_GENERIC, 0)
        except Exception as e:
            if getattr(e, "args", (None,))[0] == _ERROR_NOT_FOUND:
                return None
            raise self._fail("read a key", e) from None
        blob = bytes(cred.get("CredentialBlob") or b"")  # pywin32 stores a str as UTF-16LE
        try:
            return blob.decode("utf-16-le")
        except UnicodeDecodeError:
            raise CredentialError("The saved key could not be read.") from None

    def write(self, target, secret):
        if len(secret.encode("utf-16-le")) > MAX_BLOB:
            raise CredentialError("That key is too long for the Windows credential store.")
        try:
            self._w.CredWrite({"Type": self._w.CRED_TYPE_GENERIC, "TargetName": target, "UserName": "parley",
                               "CredentialBlob": secret, "Persist": self._w.CRED_PERSIST_LOCAL_MACHINE}, 0)
        except Exception as e:
            raise self._fail("save a key", e) from None

    def delete(self, target):
        try:
            self._w.CredDelete(target, self._w.CRED_TYPE_GENERIC, 0)
            return True
        except Exception as e:
            if getattr(e, "args", (None,))[0] == _ERROR_NOT_FOUND:
                return False
            raise self._fail("delete a key", e) from None


class UnavailableBackend:
    def read(self, target):
        return None

    def write(self, target, secret):
        raise CredentialError("Saving keys needs the Windows credential store, which is not available here.")

    def delete(self, target):
        return False


def home_tag(home: str | None = None) -> str:
    """8 hex digits identifying PARLEY_HOME ('' when it is not set, i.e. the real home)."""
    home = os.environ.get("PARLEY_HOME") if home is None else home
    if not home:
        return ""
    norm = os.path.normcase(os.path.abspath(home))
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()[:8]


def default_prefix() -> str:
    tag = home_tag()
    return f"Parley-{tag}" if tag else "Parley"


def check_key_shape(value) -> bool:
    """Printable ASCII without spaces, 1..512 characters (the value is never reported)."""
    return (isinstance(value, str) and 0 < len(value) <= MAX_KEY_CHARS and value.isascii()
            and all(32 < ord(c) < 127 for c in value))


class CredStore:
    def __init__(self, backend: Backend | None = None, prefix: str | None = None):
        if backend is None:
            backend = UnavailableBackend()
            if sys.platform == "win32":
                try:
                    backend = WinBackend()
                except ImportError:
                    pass
        self.backend, self.available = backend, not isinstance(backend, UnavailableBackend)
        self._prefix = prefix

    def target(self, backend_name: str) -> str:
        if backend_name not in KEY_BACKENDS:
            raise CredentialError(f"No stored key for backend {backend_name!r}.")
        return f"{self._prefix if self._prefix is not None else default_prefix()}/{backend_name}"

    def get(self, backend_name: str) -> str | None:
        return self.backend.read(self.target(backend_name)) or None

    def set(self, backend_name: str, key: str) -> None:
        if not check_key_shape(key):
            raise CredentialError("That doesn't look like an API key.")
        self.backend.write(self.target(backend_name), key)

    def delete(self, backend_name: str) -> bool:
        return self.backend.delete(self.target(backend_name))

    def exists(self, backend_name: str) -> bool:
        return self.get(backend_name) is not None


_default: CredStore | None = None


def default_store() -> CredStore:
    global _default
    if _default is None:
        _default = CredStore()
    return _default


def get(backend_name: str) -> str | None:
    return default_store().get(backend_name)


def set(backend_name: str, key: str) -> None:  # noqa: A001 (the API reads credstore.set / get / delete / exists)
    default_store().set(backend_name, key)


def delete(backend_name: str) -> bool:
    return default_store().delete(backend_name)


def exists(backend_name: str) -> bool:
    return default_store().exists(backend_name)


# -- lookup ----------------------------------------------------------------------------------------------------

def lookup_api_key(backend_name: str, cfg, store: CredStore | None = None) -> tuple[str, str | None]:
    """(key, source) with stored key > environment variable > legacy llm.api_key; ("", None) when there is none."""
    store = store or default_store()
    try:
        stored = store.get(backend_name) if backend_name in KEY_BACKENDS else None
    except CredentialError as e:
        log.warning("Could not read the saved %s key: %s", backend_name, e)
        stored = None
    if stored:
        return stored, SOURCE_STORE
    env = os.environ.get(ENV_VARS.get(backend_name, ""), "") if backend_name in ENV_VARS else ""
    if env:
        return env, SOURCE_ENV
    legacy = getattr(cfg, "api_key", "") if cfg is not None else ""
    if legacy:
        return legacy, SOURCE_CONFIG
    return "", None


def resolve_api_key(backend_name: str, cfg, store: CredStore | None = None) -> str:
    """The key to use for a backend ("" if none). Logs only which source supplied it."""
    key, source = lookup_api_key(backend_name, cfg, store)
    if key:
        register_secret(key)
        log.info("Using the %s API key from the %s", backend_name, source)
    return key


def warn_if_custom_endpoint(backend_name: str, base_url: str, loopback_ok: bool = False) -> None:
    """Log (host only) when a key is about to be sent somewhere other than the provider's own endpoint."""
    try:
        host = urlparse(base_url).hostname or ""
    except ValueError:
        host = ""
    if not host or host == DEFAULT_HOSTS.get(backend_name):
        return
    if loopback_ok and host in ("127.0.0.1", "localhost", "::1"):
        return
    log.warning("The %s API key will be sent to %s, not the provider's own address", backend_name, host)


# -- keeping keys out of logs ----------------------------------------------------------------------------------

_secrets: dict[str, None] = {}  # (a dict: this module defines its own set())
_KEY_SHAPE = re.compile(r"(sk-[A-Za-z0-9_-]{8,}|Bearer\s+[A-Za-z0-9._~+/=-]{8,}|x-api-key['\"]?\s*[:=]\s*['\"]?[^\s'\",}]+)", re.I)


def register_secret(value: str) -> None:
    if value and len(value) >= 6:
        _secrets[value] = None


def redact(text: str) -> str:
    for s in _secrets:
        text = text.replace(s, "[key]")
    return _KEY_SHAPE.sub("[key]", text)


class RedactFilter(logging.Filter):
    """Scrubs keys from the message and the traceback text before any handler formats the record."""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            msg = record.getMessage()
            if record.exc_info and not record.exc_text:
                record.exc_text = logging.Formatter().formatException(record.exc_info)
            if record.exc_text:
                record.exc_text = redact(record.exc_text)
            red = redact(msg)
            if red != msg:
                record.msg, record.args = red, None
        except Exception:
            record.msg, record.args, record.exc_info, record.exc_text = "[log line withheld]", None, None, None
        return True


# -- migration of a plaintext llm.api_key ------------------------------------------------------------------------

MIGRATED, NOTHING, LEFT_IN_PLACE, FAILED = "migrated", "nothing", "left-in-place", "failed"
_warned: dict[str, None] = {}


def _warn_once(tag: str, msg: str, *args) -> None:
    if tag not in _warned:
        _warned[tag] = None
        log.warning(msg, *args)


def _legacy_key(text: str) -> str | None:
    llm = tomllib.loads(text).get("llm")
    key = llm.get("api_key") if isinstance(llm, dict) else None
    return key.strip() if isinstance(key, str) and key.strip() else None


def _target_backend(configured: str, key: str) -> str | None:
    if configured in KEY_BACKENDS:
        return configured
    if configured == "server":  # a key for your own server is not an Anthropic / OpenAI key whatever its prefix
        return None
    if key.startswith("sk-ant-"):
        return "anthropic"
    if key.startswith("sk-"):
        return "openai"
    return None


def migrate_legacy_key(path: Path, store: CredStore | None = None) -> str:
    """Move a plaintext llm.api_key from config.toml into the credential store. Idempotent; never raises.

    Order: write to the store, read it back and compare, only then remove the line from config.toml, and finally
    delete a config.toml.bak that held the key. On any failure config.toml is left exactly as it was."""
    path = Path(path)
    try:
        return _migrate(path, store or default_store())
    except Exception as e:  # never stop startup; the message holds only the exception type
        _warn_once("migrate-error", "API key migration failed (%s); config.toml was left as it was", type(e).__name__)
        return FAILED


def _migrate(path: Path, store: CredStore) -> str:
    from . import config_io

    if not path.is_file():
        return NOTHING
    try:
        text = path.read_bytes().decode("utf-8")
        key = _legacy_key(text)
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError):
        _warn_once("unreadable", "Could not check %s for an API key (unreadable or has a mistake); not touching it", path.name)
        return FAILED
    if key is None:
        return NOTHING
    llm = tomllib.loads(text).get("llm", {})
    backend = _target_backend(str(llm.get("backend", "")), key)
    if backend is None:
        _warn_once("ambiguous", "llm.api_key in %s can't be matched to Anthropic or OpenAI; it was left in place. "
                   "Save it with: parley --set-key <anthropic|openai>, then delete the line", path.name)
        return LEFT_IN_PLACE
    if not check_key_shape(key):
        _warn_once("shape", "llm.api_key in %s is not a usable key; it was left in place", path.name)
        return LEFT_IN_PLACE
    if not store.available:
        _warn_once("unavailable", "The credential store is not available here; llm.api_key was left in %s", path.name)
        return LEFT_IN_PLACE
    try:  # the file must be writable before the key moves
        with open(path, "r+b"):
            pass
    except OSError:
        _warn_once("locked", "%s is read-only or locked; llm.api_key was left in place", path.name)
        return FAILED

    existing = store.get(backend)
    if existing is not None and existing != key:
        _warn_once("differs", "A different %s key is already saved, so llm.api_key in %s was left in place "
                   "(delete that line if it is out of date)", backend, path.name)
        return LEFT_IN_PLACE
    wrote = existing is None
    try:
        if wrote:
            store.set(backend, key)
        if store.get(backend) != key:
            raise CredentialError("read-back mismatch")
        config_io.save_changes(path, {"llm.api_key": None})
        if _legacy_key(path.read_bytes().decode("utf-8")) is not None:
            raise config_io.ConfigFileError("key still present")
    except Exception as e:
        if wrote:
            try:
                store.delete(backend)  # nothing else held this entry, so undo the half step
            except Exception:
                pass
        _warn_once("failed", "Could not move llm.api_key into the credential store (%s); %s was left as it was",
                   type(e).__name__, path.name)
        return FAILED

    bak = path.with_name(path.name + ".bak")
    try:
        if bak.is_file() and (key in bak.read_text(encoding="utf-8", errors="replace") or _bak_has_key(bak)):
            bak.unlink()
    except OSError:
        log.warning("Could not delete %s, which may still hold the old API key; delete it by hand", bak.name)
    log.info("Moved the %s API key from %s into the Windows credential store. If this file was ever synced or "
             "shared, rotate the key.", backend, path.name)
    return MIGRATED


def _bak_has_key(bak: Path) -> bool:
    try:
        return _legacy_key(bak.read_text(encoding="utf-8")) is not None
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError):
        return False


# -- command line --------------------------------------------------------------------------------------------------

UnavailableMessage = "Saving keys needs the Windows credential store, which is not available here. Use the environment variable instead."


def cli_set_key(backend_name: str, store: CredStore | None = None, out=None, err=None) -> int:
    import getpass

    out, err = out or sys.stdout, err or sys.stderr
    store = store or default_store()
    if not store.available:
        print(UnavailableMessage, file=err)
        return 1
    key = getpass.getpass(f"{backend_name} API key (input is hidden): ").strip()
    if not key:
        print("No key entered; nothing saved.", file=err)
        return 1
    if not check_key_shape(key):
        print("That doesn't look like an API key (letters and digits only, no spaces); nothing saved.", file=err)
        return 1
    register_secret(key)
    try:
        store.set(backend_name, key)
        if store.get(backend_name) != key:
            raise CredentialError("The saved key did not read back correctly.")
    except CredentialError as e:
        print(f"Not saved: {e}", file=err)
        return 1
    print(f"Saved the {backend_name} API key in the Windows credential store.", file=out)
    return 0


def cli_delete_key(backend_name: str, store: CredStore | None = None, out=None, err=None) -> int:
    out, err = out or sys.stdout, err or sys.stderr
    try:
        removed = (store or default_store()).delete(backend_name)
    except CredentialError as e:
        print(f"Not deleted: {e}", file=err)
        return 1
    print(f"Deleted the saved {backend_name} key." if removed else f"No saved {backend_name} key.", file=out)
    return 0


def cli_key_status(cfg, store: CredStore | None = None, out=None) -> int:
    """Prints only Saved / Not saved per backend and which source is in use; never a key, prefix or length."""
    import dataclasses

    out = out or sys.stdout
    store = store or default_store()
    for name in KEY_BACKENDS:
        try:
            saved = "Saved" if store.exists(name) else "Not saved"
        except CredentialError:
            saved = "Unknown (credential store error)"
        llm = cfg.llm if cfg.llm.backend == name else dataclasses.replace(cfg.llm, api_key="")
        _, source = lookup_api_key(name, llm, store)
        print(f"{name}: {saved}; in use: {source or 'none'}", file=out)
    return 0
