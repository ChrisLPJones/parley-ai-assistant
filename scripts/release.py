"""Turn a clean `git archive` export of a ref into a publishable Parley release (never the working tree).

    uv run python scripts/release.py --notes "Fixes X" --dry-run                 # build, print the exact ssh/scp commands
    uv run python scripts/release.py --notes-file RELEASE_NOTES                  # build and publish to updates-deploy
    uv run python scripts/release.py --notes "..." --target D:\\some\\dir         # publish to a local directory (tests)

Steps: export the ref (default origin/main) -> `uv sync --frozen --group build` in the export with a scrubbed PATH (no
CUDA toolkit) -> scripts/build.py (also regenerates cuda_pack.json; it must equal the committed one) -> scripts/make_installer.py
-> local smoke (`--version`, `--self-check`) on a throwaway PARLEY_HOME -> server layout:

    blobs/<sha256>                    every file of the app tree, content addressed (shared by all versions)
    <ver>/windows-x64/files.json      the per-version index (same bytes as the tree's .parley-index.json)
    Parley-Setup-<ver>.exe            the installer
    version.json                      the manifest, uploaded LAST, atomically; the old one is kept as version.json.prev

The version comes only from `__version__` in the exported source (bumping it is a normal PR). The script never pushes or tags: it
prints the `git tag` command for the coordinator. It refuses: `--allow-rejected-paths` builds, an unknown ref, an ssh publish of a ref
that is not on a freshly fetched origin/main, a version not greater than the published one (unless --force-republish), a feed URL that is not https
(the localhost-http escape exists for local test targets only), and a tag that already points at another commit.
Rollback = fix forward with a higher version, or restore version.json.prev before clients update.
"""
import argparse
import datetime
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

import httpx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from parley import launcher  # noqa: E402
from parley.update import (MAX_MANIFEST_BYTES, PLATFORM_KEY, UpdateError, Updater, parse_index, parse_manifest,  # noqa: E402
                           validate_index_files)
from parley.version import compare_versions, parse_version  # noqa: E402

DEFAULT_REF = "origin/main"
DEFAULT_FEED = "https://updates.chrislpjones.com/parley/version.json"
DEFAULT_HOST = "updates-deploy"
DEFAULT_REMOTE_ROOT = "/srv/updates/parley"
MAX_NOTES = 1500  # the client's prompt shows at most this much
MAX_NOTES_FILE = 64 * 1024
LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}
SSH_OPTS = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "-o", "StrictHostKeyChecking=yes"]
_HEX64 = re.compile(r"[0-9a-f]{64}", re.ASCII)
_HOST = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,62}", re.ASCII)
_REMOTE_ROOT = re.compile(r"/[A-Za-z0-9._/-]{1,200}", re.ASCII)
_VERSION_LINE = re.compile(r'^__version__\s*=\s*"([^"\n]*)"\s*$', re.MULTILINE)


class ReleaseError(Exception):
    pass


# ---------- ref, export, version ----------

def git(*args: str, repo: Path = ROOT) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True)


def fetch_origin_main(repo: Path = ROOT) -> None:
    """A real publish must see the current origin/main: a stale local copy could make an old commit look merged."""
    r = git("fetch", "origin", "main", repo=repo)
    if r.returncode != 0:
        raise ReleaseError(f"git fetch origin main failed, so origin/main may be stale: {r.stderr.strip()[-200:]}")


def resolve_ref(ref: str, repo: Path = ROOT) -> str:
    """Commit sha of `ref`, or ReleaseError (unknown ref, or something that is not a commit)."""
    if not ref or ref.startswith("-") or any(c.isspace() or ord(c) < 32 for c in ref):
        raise ReleaseError(f"bad ref {ref!r}")
    r = git("rev-parse", "--verify", "--quiet", "--end-of-options", f"{ref}^{{commit}}", repo=repo)
    sha = r.stdout.strip()
    if r.returncode != 0 or not re.fullmatch(r"[0-9a-f]{40}", sha):
        raise ReleaseError(f"unknown ref {ref!r} (not a commit in this repository)")
    return sha


def require_on_main(sha: str, main_ref: str = DEFAULT_REF, repo: Path = ROOT) -> None:
    """A real publish only ships what is on origin/main (the protected branch), nothing from a feature branch."""
    main = resolve_ref(main_ref, repo)
    if git("merge-base", "--is-ancestor", sha, main, repo=repo).returncode != 0:
        raise ReleaseError(f"{sha[:10]} is not on {main_ref}: publish only merged commits (use --target DIR for tests)")


def export_ref(sha: str, dest: Path, repo: Path = ROOT) -> None:
    """Extract `git archive sha` into the empty dir `dest`: tracked files of that commit only."""
    dest.mkdir(parents=True, exist_ok=True)
    if any(dest.iterdir()):
        raise ReleaseError(f"{dest} is not empty")
    p = subprocess.Popen(["git", "-C", str(repo), "archive", "--format=tar", sha], stdout=subprocess.PIPE)
    try:
        with tarfile.open(fileobj=p.stdout, mode="r|") as tf:
            tf.extractall(dest, filter="data")
    finally:
        p.stdout.close()
        rc = p.wait()
    if rc != 0:
        raise ReleaseError("git archive failed")


def read_version(export: Path) -> str:
    """`__version__` from the exported source (text only, nothing is imported or run)."""
    text = (export / "src" / "parley" / "__init__.py").read_text("utf-8")
    m = _VERSION_LINE.search(text)
    if not m:
        raise ReleaseError("no __version__ line in src/parley/__init__.py")
    try:
        parse_version(m.group(1))
    except ValueError as e:
        raise ReleaseError(f"__version__ {m.group(1)!r} is not MAJOR.MINOR.PATCH") from e
    return m.group(1)


def override_version(export: Path, version: str) -> None:
    """Test/acceptance only: rewrite `__version__` in the EXPORT (never the working tree, never committed)."""
    parse_version(version)
    path = export / "src" / "parley" / "__init__.py"
    text = path.read_text("utf-8")
    new, n = _VERSION_LINE.subn(f'__version__ = "{version}"', text, count=1)
    if n != 1:
        raise ReleaseError("no __version__ line to override")
    path.write_text(new, "utf-8")


def check_tag(version: str, sha: str, repo: Path = ROOT) -> None:
    r = git("rev-parse", "-q", "--verify", f"refs/tags/v{version}^{{commit}}", repo=repo)
    if r.returncode == 0 and r.stdout.strip() != sha:
        raise ReleaseError(f"tag v{version} already exists and points at another commit ({r.stdout.strip()[:10]})")


def tag_command(version: str, sha: str) -> list[str]:
    return ["git", "tag", f"v{version}", sha]


# ---------- inputs: notes, feed URL ----------

def clean_notes(text: str) -> str:
    """Plain text for the update prompt: CRLF -> LF, control characters dropped (newline and tab stay), trimmed, capped."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "".join(c for c in text if c in "\n\t" or unicodedata.category(c)[0] != "C").strip()
    if len(text) > MAX_NOTES:
        raise ReleaseError(f"release notes are {len(text)} characters; keep them under {MAX_NOTES}")
    return text


def read_notes(notes: str | None, notes_file: str | None) -> str:
    if notes is not None and notes_file is not None:
        raise ReleaseError("give --notes or --notes-file, not both")
    if notes_file is not None:
        p = Path(notes_file)
        try:
            if p.stat().st_size > MAX_NOTES_FILE:
                raise ReleaseError(f"{p} is larger than {MAX_NOTES_FILE // 1024} KB")
            notes = p.read_text("utf-8")
        except (OSError, UnicodeDecodeError) as e:
            raise ReleaseError(f"cannot read notes file {p}: {e}") from e
    return clean_notes(notes or "")


def check_feed_url(url: str, *, test_http_local: bool = False) -> str:
    """The feed URL baked into the build must be one the shipped client accepts: https. Only `test_http_local` (local
    test targets) allows http, and only to localhost."""
    try:
        Updater(url, "0.0.0", _allow_http_localhost=test_http_local)  # the real client's own URL rules
    except UpdateError as e:
        raise ReleaseError(f"feed URL {url!r} is not acceptable: {e} (published builds need https)") from e
    u = urlsplit(url)
    if u.scheme != "https" and not (test_http_local and u.hostname in LOCAL_HOSTS):
        raise ReleaseError(f"feed URL {url!r} must be https")
    if not u.path.endswith("/version.json"):
        raise ReleaseError("feed URL must end in /version.json")
    return url


# ---------- build ----------

def clean_env(env: dict[str, str] | None = None) -> dict[str, str]:
    """Environment for the build: PATH without CUDA toolkit entries (PyInstaller resolves DLLs from PATH), no CUDA_*, no
    inherited venv. A release must come from a clean environment (PROJECT.md Gotchas)."""
    env = dict(os.environ if env is None else env)
    for k in [k for k in env if k.upper().startswith("CUDA") or k.upper() in ("VIRTUAL_ENV", "PYTHONPATH", "PYTHONHOME",
                                                                               "UV_PROJECT_ENVIRONMENT")]:
        del env[k]
    keep = [d for d in env.get("PATH", "").split(os.pathsep)
            if d and "cuda" not in d.lower() and "nvidia gpu computing toolkit" not in d.lower()]
    env["PATH"] = os.pathsep.join(keep)
    return env


def build_commands(feed_url: str) -> list[list[str]]:
    """The commands run inside the export, in order (list args, no shell)."""
    return [["uv", "sync", "--frozen", "--group", "build"],
            ["uv", "run", "--frozen", "--no-sync", "--group", "build", "python", "scripts/build.py", "--feed-url", feed_url],
            ["uv", "run", "--frozen", "--no-sync", "--group", "build", "python", "scripts/make_installer.py"]]


def run_build(export: Path, feed_url: str, env: dict[str, str] | None = None) -> None:
    env = clean_env(env)
    for cmd in build_commands(feed_url):
        print("+", shlex.join(cmd), flush=True)
        if subprocess.run(cmd, cwd=export, env=env).returncode != 0:
            raise ReleaseError(f"build step failed: {shlex.join(cmd[:4])} ...")


def check_cuda_manifest(export: Path, sha: str, repo: Path = ROOT) -> None:
    """build.py regenerates src/parley/cuda_pack.json from uv.lock; the committed copy must already be that."""
    r = subprocess.run(["git", "-C", str(repo), "show", f"{sha}:src/parley/cuda_pack.json"], capture_output=True)
    if r.returncode != 0 or r.stdout.replace(b"\r\n", b"\n") != (export / "src" / "parley" / "cuda_pack.json").read_bytes().replace(b"\r\n", b"\n"):
        raise ReleaseError("cuda_pack.json in the ref differs from what uv.lock implies: regenerate it, merge, release again")


def smoke_home_ok(home: Path) -> bool:
    """Smoke runs only on a throwaway store named parley_test_*, never the real Parley home."""
    real = os.environ.get("LOCALAPPDATA")
    if real and home.resolve() == (Path(real) / "parley").resolve():
        return False
    return home.name.startswith("parley_test_")


def smoke(appdir: Path, version: str, home: Path, ask: bool = False) -> None:
    if not smoke_home_ok(home):
        raise ReleaseError(f"smoke home {home} must be a throwaway dir named parley_test_<topic>")
    home.mkdir(parents=True, exist_ok=True)
    env = {**clean_env(), "PARLEY_HOME": str(home)}
    print(f"smoke: PARLEY_HOME={home}", flush=True)
    exe = str(appdir / launcher.CHILD_EXE)
    r = subprocess.run([exe, "--version"], env=env, capture_output=True, text=True, timeout=120)
    if r.returncode != 0 or r.stdout.strip() != f"parley {version}":
        raise ReleaseError(f"smoke --version: exit {r.returncode}, output {r.stdout.strip()!r}")
    steps = [["--self-check"]] + ([["--ask", "What is the capital of Peru?"]] if ask else [])
    for step in steps:
        r = subprocess.run([exe, *step], env=env, cwd=appdir, capture_output=True, text=True, timeout=1800)
        if r.returncode != 0:
            raise ReleaseError(f"smoke {step[0]} failed (exit {r.returncode}): {(r.stderr or r.stdout).strip()[-300:]}")
        print(f"smoke {step[0]}: ok", flush=True)


# ---------- layout ----------

@dataclass
class Layout:
    version: str
    blobs: dict[str, Path]  # sha256 -> source file in the built app tree
    files_json_path: str  # relative path on the server
    files_json: bytes
    installer: Path
    installer_sha256: str
    version_json: bytes
    installer_size: int = 0
    blob_bytes: int = 0
    manifest: dict = field(default_factory=dict)

    @property
    def installer_name(self) -> str:
        return self.installer.name


def make_manifest(version: str, notes: str, installer: Path, installer_sha: str, index_path: str, index_sha: str,
                  released: str) -> dict:
    entry = {"file": installer.name, "sha256": installer_sha, "size": installer.stat().st_size,
             "index": index_path, "index_sha256": index_sha}
    return {"schema": 1, "version": version, "notes": notes, "released": released,
            "file": installer.name, "sha256": installer_sha,
            "platforms": {PLATFORM_KEY: entry}, "signature": None}


def make_layout(install_tree: Path, installer: Path, version: str, notes: str, released: str | None = None) -> Layout:
    """Verify the built tree and compute everything to publish. Raises ReleaseError for anything the client would reject."""
    ptr = launcher.read_pointer(install_tree / launcher.POINTER)
    if ptr is None or ptr.version != version or not ptr.index_sha256:
        raise ReleaseError(f"{install_tree} is not a build of {version}")
    appdir = launcher.app_path(install_tree, version)
    try:
        entries = launcher.read_index_entries(appdir, ptr.index_sha256)
        validate_index_files([{"path": p, "sha256": h, "size": n} for p, h, n in entries])
        launcher.verify_tree(appdir, entries)
    except (UpdateError, launcher.LauncherError, OSError, ValueError, KeyError, TypeError) as e:
        # includes paths the update client rejects (a --allow-rejected-paths build can never be released)
        raise ReleaseError(f"app tree is not publishable: {e}") from e
    if installer.name != f"Parley-Setup-{version}.exe" or not installer.is_file():
        raise ReleaseError(f"installer {installer} is missing or not Parley-Setup-{version}.exe")
    files_json = (appdir / launcher.INDEX_NAME).read_bytes()
    index_sha = hashlib.sha256(files_json).hexdigest()
    installer_sha = launcher.sha256_file(installer)
    index_path = f"{version}/{PLATFORM_KEY}/files.json"
    released = released or datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")
    manifest = make_manifest(version, notes, installer, installer_sha, index_path, index_sha, released)
    version_json = json.dumps(manifest, indent=1).encode("utf-8") + b"\n"
    try:  # exactly what the shipped client will do with it
        m = parse_manifest(version_json)
        parse_index(files_json, index_sha, version)
    except UpdateError as e:
        raise ReleaseError(f"the client would reject this release: {e}") from e
    if m.version != version or m.platform.sha256 != installer_sha:
        raise ReleaseError("manifest does not round-trip")
    blobs = {h: appdir / p for p, h, _n in sorted(entries, reverse=True)}
    return Layout(version, blobs, index_path, files_json, installer, installer_sha, version_json,
                  installer.stat().st_size, sum(n for _p, _h, n in entries), manifest)


# ---------- publish: version check ----------

def parse_published_version(raw: bytes | None, force: bool = False) -> str | None:
    """Version in a published version.json, None if there is none. An unreadable one is an error (force-republish skips)."""
    if raw is None:
        return None
    try:
        v = json.loads(raw.decode("utf-8"))["version"]
        parse_version(v)
    except (ValueError, KeyError, TypeError, AttributeError) as e:  # UnicodeDecodeError is a ValueError
        if force:
            print(f"WARNING: the published version.json is unreadable ({e}); --force-republish replaces it", file=sys.stderr)
            return None
        raise ReleaseError(f"the published version.json is unreadable ({e}); fix it by hand or use --force-republish") from e
    return v


def check_newer(new: str, published: str | None, force: bool) -> None:
    if published is None:
        return
    c = compare_versions(new, published)
    if c > 0:
        return
    if not force:
        what = "equal to" if c == 0 else "older than"
        raise ReleaseError(f"version {new} is {what} the published {published}: bump __version__ through a PR "
                           "(--force-republish overwrites; clients that already saw a higher version will ignore it)")
    print(f"WARNING: --force-republish: {new} replaces published {published}", file=sys.stderr)


# ---------- publish: local directory ----------

def _atomic_copy(src: Path, dst: Path) -> None:
    tmp = dst.with_name(f".{dst.name}.tmp")
    shutil.copyfile(src, tmp)
    os.replace(tmp, dst)


def _atomic_write(dst: Path, data: bytes) -> None:
    tmp = dst.with_name(f".{dst.name}.tmp")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, dst)


def read_local_version(target: Path, force: bool = False) -> str | None:
    p = target / "version.json"
    return parse_published_version(p.read_bytes() if p.is_file() else None, force)


def publish_local(layout: Layout, target: Path, on_step: Callable[[str], None] | None = None) -> None:
    """Write the layout under `target`. Order: blobs, installer, files.json, then version.json (atomic rename; the previous
    one is first copied to version.json.prev). A reader never sees a version.json whose files are not all in place."""
    step = on_step or (lambda _s: None)
    (target / "blobs").mkdir(parents=True, exist_ok=True)
    for sha, src in layout.blobs.items():
        dst = target / "blobs" / sha
        if dst.is_file() and dst.stat().st_size == src.stat().st_size:
            continue  # content addressed and written atomically: an existing blob is complete
        _atomic_copy(src, dst)
        step(f"blobs/{sha}")
    _atomic_copy(layout.installer, target / layout.installer_name)
    step(layout.installer_name)
    idx = target / layout.files_json_path
    idx.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write(idx, layout.files_json)
    step(layout.files_json_path)
    live = target / "version.json"
    if live.is_file():
        _atomic_copy(live, target / "version.json.prev")
        step("version.json.prev")
    _atomic_write(live, layout.version_json)
    step("version.json")


# ---------- publish: ssh ----------

def check_ssh_target(host: str, root: str) -> None:
    if not _HOST.fullmatch(host):
        raise ReleaseError(f"bad ssh host alias {host!r}")
    if not _REMOTE_ROOT.fullmatch(root) or ".." in root.split("/") or root.endswith("/") and root != "/":
        raise ReleaseError(f"bad remote directory {root!r}")


def remote(script: str, *args: str) -> str:
    """One remote command string for ssh: `sh -c SCRIPT sh ARGS...`, every piece shell-quoted."""
    return shlex.join(["sh", "-c", script, "sh", *args])


# Each script starts with umask 022 (files and dirs the scripts create are world-readable, never group-writable).
# Each script gets $1 = remote root, $2 = version.
REMOTE_READ_VERSION = 'umask 022; if [ -f "$1/version.json" ]; then cat "$1/version.json"; fi'
REMOTE_PREPARE = ('umask 022; set -eu; mkdir -p "$1/blobs" "$1/$2/windows-x64"; rm -rf "$1/incoming-$2"; '
                  'ls -1 "$1/blobs"')
REMOTE_MOVE = ('umask 022; set -eu; root=$1; inc="$1/incoming-$2"; chmod -R u=rwX,go=rX "$inc"; '
               'if [ -d "$inc/blobs" ]; then for f in "$inc"/blobs/*; do [ -e "$f" ] && mv -f "$f" "$root/blobs/"; done; fi; '
               'mkdir -p "$root/$2/windows-x64"; '
               'mv -f "$inc/$2/windows-x64/files.json" "$root/$2/windows-x64/files.json"; '
               'mv -f "$inc/Parley-Setup-$2.exe" "$root/Parley-Setup-$2.exe"; rm -rf "$inc"')
REMOTE_COMMIT = ('umask 022; set -eu; cd "$1"; chmod 644 version.json.tmp; '
                 'if [ -f version.json ]; then cp -p version.json version.json.prev; fi; mv -f version.json.tmp version.json')


@dataclass
class Command:
    argv: list[str]
    why: str
    cwd: Path | None = None

    def shown(self) -> str:
        return shlex.join(self.argv)


def stage_upload(layout: Layout, stage: Path, missing_blobs: list[str]) -> Path:
    """Build `stage/incoming-<ver>/` (hardlinks where possible) plus `stage/version.json.tmp`; returns the incoming dir."""
    inc = stage / f"incoming-{layout.version}"
    (inc / "blobs").mkdir(parents=True)

    def put(src: Path, dst: Path) -> None:
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(src, dst)
        except OSError:
            shutil.copyfile(src, dst)

    for sha in missing_blobs:
        put(layout.blobs[sha], inc / "blobs" / sha)
    put(layout.installer, inc / layout.installer_name)
    idx = inc / layout.files_json_path
    idx.parent.mkdir(parents=True, exist_ok=True)
    idx.write_bytes(layout.files_json)
    (stage / "version.json.tmp").write_bytes(layout.version_json)
    return inc


def ssh_commands(version: str, host: str, root: str, stage: Path) -> list[Command]:
    """The upload commands in order (list args, no shell locally; remote scripts get their values as positional args).
    1 prepare+list blobs, 2 scp the incoming dir, 3 move it into place, 4 scp version.json.tmp, 5 commit (atomic mv)."""
    check_ssh_target(host, root)
    parse_version(version)
    ssh = ["ssh", *SSH_OPTS, host]
    return [
        Command([*ssh, remote(REMOTE_PREPARE, root, version)], "create dirs, clear a stale incoming dir, list existing blobs"),
        Command(["scp", "-r", *SSH_OPTS, f"incoming-{version}", f"{host}:{root}/"], "upload new blobs, installer, files.json", stage),
        Command([*ssh, remote(REMOTE_MOVE, root, version)], "move blobs/installer/files.json into place (chmod go=rX)"),
        Command(["scp", *SSH_OPTS, "version.json.tmp", f"{host}:{root}/version.json.tmp"], "upload the manifest (still not live)", stage),
        Command([*ssh, remote(REMOTE_COMMIT, root, version)], "keep version.json.prev, atomic mv version.json (LAST)"),
    ]


def read_ssh_version(host: str, root: str, force: bool = False) -> str | None:
    check_ssh_target(host, root)
    r = subprocess.run(["ssh", *SSH_OPTS, host, remote(REMOTE_READ_VERSION, root)], capture_output=True, timeout=60)
    if r.returncode != 0:
        raise ReleaseError(f"cannot read the published version over ssh (exit {r.returncode}): {r.stderr.decode(errors='replace').strip()[-200:]}")
    return parse_published_version(r.stdout or None, force)


def publish_ssh(layout: Layout, host: str, root: str, work: Path, *, dry_run: bool, runner=subprocess.run) -> None:
    """Upload. With dry_run nothing is run and no connection is made: the exact commands are printed (without a server
    listing every blob is assumed missing, so the real run uploads fewer)."""
    stage = work / "upload"
    stage.mkdir(parents=True, exist_ok=True)
    cmds = ssh_commands(layout.version, host, root, stage)
    if dry_run:
        stage_upload(layout, stage, [])  # small files only; the blob list is only known after command 1
        print(f"DRY RUN: nothing is run. Staging dir: {stage}")
        print(f"{len(layout.blobs)} blobs ({layout.blob_bytes / 1e6:.0f} MB) would be checked against the server's listing.")
        for i, c in enumerate(cmds, 1):
            print(f"[{i}] {c.why}\n    cwd={c.cwd or '.'}\n    {c.shown()}")
        return
    existing: set[str] = set()
    for i, c in enumerate(cmds, 1):
        print(f"[{i}/{len(cmds)}] {c.why}", flush=True)
        if i == 2:
            missing = [s for s in layout.blobs if s not in existing]
            print(f"    {len(missing)} of {len(layout.blobs)} blobs are new", flush=True)
            stage_upload(layout, stage, missing)
        r = runner(c.argv, cwd=c.cwd, capture_output=(i == 1), text=(i == 1))
        if r.returncode != 0:
            raise ReleaseError(f"command {i} failed (exit {r.returncode}): {c.shown()[:200]}")
        if i == 1:
            existing = {ln.strip() for ln in r.stdout.splitlines() if _HEX64.fullmatch(ln.strip())}


# ---------- post-publish verification ----------

def verify_instruction(url: str, version_json: bytes) -> str:
    sha = hashlib.sha256(version_json).hexdigest()
    code = "import hashlib,sys,urllib.request as u;print(hashlib.sha256(u.urlopen(sys.argv[1]).read()).hexdigest())"
    cmd = shlex.join(["python", "-c", code, url])  # the URL travels as an argument, never inside the code
    lines = [f"Verify the live manifest. URL: {url}", f"Expected sha256: {sha}", f"  {cmd}",
             "or pass --verify-url on the next release run."]
    return "\n".join(lines)


def verify_published(url: str, version_json: bytes, *, _allow_http_localhost: bool = False) -> None:
    """Fetch the live version.json with the shipped update client's own HTTP code (https, certificate checks, same-origin
    redirects only) and require the exact bytes just published. Raises ReleaseError loudly on any difference."""
    try:
        client = Updater(url, "0.0.0", _allow_http_localhost=_allow_http_localhost)
        got = client._get_bytes(url, MAX_MANIFEST_BYTES)
    except (UpdateError, httpx.HTTPError, OSError) as e:
        raise ReleaseError(f"VERIFY FAILED: could not fetch {url}: {e}") from e
    if hashlib.sha256(got).digest() != hashlib.sha256(version_json).digest():
        raise ReleaseError(f"VERIFY FAILED: {url} serves sha256 {hashlib.sha256(got).hexdigest()}, "
                           f"published {hashlib.sha256(version_json).hexdigest()}")
    print(f"verified: {url} serves the published version.json (sha256 {hashlib.sha256(got).hexdigest()})", flush=True)


# ---------- orchestration ----------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ref", default=DEFAULT_REF, help=f"git ref to export and build (default {DEFAULT_REF}; never the working tree)")
    ap.add_argument("--notes", help="release notes (plain text, short)")
    ap.add_argument("--notes-file", help="read the release notes from this file (e.g. RELEASE_NOTES)")
    ap.add_argument("--feed-url", default=DEFAULT_FEED, help="update feed URL baked into the build (https)")
    ap.add_argument("--target", help="publish to this local directory instead of over ssh (used by tests)")
    ap.add_argument("--ssh-host", default=DEFAULT_HOST, help="ssh alias to publish to")
    ap.add_argument("--remote-dir", default=DEFAULT_REMOTE_ROOT, help="directory on the server")
    ap.add_argument("--verify-url", help="after publishing, fetch this https version.json with the update client's code and "
                    "require it to equal the published manifest (fails loudly); without it the exact manual check is printed")
    ap.add_argument("--dry-run", action="store_true", help="build, then print the exact publish commands and run none")
    ap.add_argument("--force-republish", action="store_true", help="allow a version equal to or older than the published one")
    ap.add_argument("--work-dir", help="empty/new dir for the export and build (default: a temp dir, removed on success)")
    ap.add_argument("--keep-work", action="store_true", help="do not remove the work dir")
    ap.add_argument("--smoke-home", help="throwaway PARLEY_HOME for the smoke run (name must start with parley_test_)")
    ap.add_argument("--smoke-ask", action="store_true", help="also run --ask in the smoke test (downloads models into the smoke home)")
    ap.add_argument("--version-override", help="TEST ONLY: replace __version__ in the export (local --target only, no tag)")
    ap.add_argument("--test-http-feed", action="store_true",
                    help="TEST ONLY: allow an http://localhost feed URL (local --target only; such a build is not shippable)")
    ap.add_argument("--allow-rejected-paths", action="store_true", help="refused: release builds never use it")
    return ap.parse_args(argv)


def validate_args(a: argparse.Namespace) -> str:
    """All refusals that need no git or build. Returns the cleaned notes."""
    if a.allow_rejected_paths:
        raise ReleaseError("--allow-rejected-paths builds can never be released (an update could not install them)")
    if (a.version_override or a.test_http_feed) and not a.target:
        raise ReleaseError("--version-override and --test-http-feed are for local --target runs only")
    if a.test_http_feed and not a.feed_url.startswith("http://"):
        raise ReleaseError("--test-http-feed is only for an http://localhost feed URL")
    check_feed_url(a.feed_url, test_http_local=bool(a.test_http_feed))
    if a.verify_url:
        check_feed_url(a.verify_url, test_http_local=bool(a.test_http_feed))
    if a.version_override:
        try:
            parse_version(a.version_override)
        except ValueError as e:
            raise ReleaseError(f"bad --version-override: {e}") from e
    if not a.target:
        check_ssh_target(a.ssh_host, a.remote_dir)
    return read_notes(a.notes, a.notes_file)


def run(argv: list[str] | None = None, export_hook: Callable[[Path], None] | None = None) -> int:
    """`export_hook(export_dir)` runs after the version override, before the build (acceptance tests use it to build a
    deliberately broken release; it only ever sees the throwaway export)."""
    a = parse_args(argv)
    notes = validate_args(a)
    if not a.target and not a.dry_run:
        fetch_origin_main()
    sha = resolve_ref(a.ref)
    if not a.target:
        require_on_main(sha)
    work_owned = a.work_dir is None
    work = Path(a.work_dir) if a.work_dir else Path(tempfile.mkdtemp(prefix=f"parley_release_{sha[:8]}_"))
    ok = False
    try:
        export = work / "export"
        export_ref(sha, export)
        if a.version_override:
            override_version(export, a.version_override)
        version = read_version(export)
        if not a.version_override:
            check_tag(version, sha)
        if export_hook:
            export_hook(export)
        target = Path(a.target) if a.target else None
        published = read_local_version(target, a.force_republish) if target else (
            None if a.dry_run else read_ssh_version(a.ssh_host, a.remote_dir, a.force_republish))
        if target is None and a.dry_run:
            print("DRY RUN: not connecting, so the published version is not compared")
        check_newer(version, published, a.force_republish)
        print(f"releasing parley {version} from {a.ref} ({sha[:10]}); feed {a.feed_url}", flush=True)
        run_build(export, a.feed_url)
        check_cuda_manifest(export, sha)
        install_tree, appdir = export / "dist" / "install", launcher.app_path(export / "dist" / "install", version)
        layout = make_layout(install_tree, export / "dist" / "installer" / f"Parley-Setup-{version}.exe", version, notes)
        home = Path(a.smoke_home) if a.smoke_home else Path(tempfile.gettempdir()) / "parley_test_release"
        smoke(appdir, version, home, a.smoke_ask)
        print(f"installer {layout.installer_name}: {layout.installer_size / 1e6:.1f} MB, sha256 {layout.installer_sha256}\n"
              f"app tree: {len(layout.blobs)} blobs, {layout.blob_bytes / 1e6:.0f} MB, files.json {len(layout.files_json)} bytes, "
              f"version.json {len(layout.version_json)} bytes", flush=True)
        if target is not None and a.dry_run:
            print(f"DRY RUN: would publish {len(layout.blobs)} blobs, installer, {layout.files_json_path}, version.json to {target}")
        elif target is not None:
            publish_local(layout, target)
            print(f"published to {target}")
        else:
            publish_ssh(layout, a.ssh_host, a.remote_dir, work, dry_run=a.dry_run)
        if not a.dry_run:
            if a.verify_url:
                try:
                    verify_published(a.verify_url, layout.version_json, _allow_http_localhost=bool(a.test_http_feed))
                except ReleaseError as e:
                    raise ReleaseError(f"{e}\nThe publish ALREADY HAPPENED: the files are live. The tag command is withheld. "
                                       "A CDN cache can serve the old manifest for a while and cause a false failure: "
                                       "re-check the URL later before treating this as a bad release.") from e
            else:
                print(verify_instruction(a.feed_url, layout.version_json))
        if not a.version_override:
            print("\nNot tagged or pushed by this script. Coordinator, after the publish succeeded:\n  " + shlex.join(tag_command(version, sha)))
        ok = True
        return 0
    finally:
        if work_owned and ok and not a.keep_work:
            shutil.rmtree(work, ignore_errors=True)
        elif not ok:
            print(f"work dir kept for diagnosis: {work}", file=sys.stderr)


def main() -> int:
    try:
        return run()
    except ReleaseError as e:
        print(f"release refused/failed: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
