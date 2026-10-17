"""Build the frozen (PyInstaller) Windows copy of Parley and lay it out the way the launcher expects.

    uv run --group build python scripts/build.py [--feed-url URL]

Result in dist/install/ (this is what the installer will ship):
    Parley.exe                  launcher (stdlib only)
    current.json                pointer to the one app version, state good
    app/<version>/              parley-app.exe, _internal/, and .parley-index.json (sha256 + size of every other file)

`src/parley/_build.py` (feed URL, platform key) is written for the duration of the build and removed afterwards; it
is gitignored and never committed. An empty feed URL (the default) leaves updates off in the built copy.
Anything the update client's path rule rejects (ASCII letters, digits, `.`, `_`, `+`, `-` per component) fails the build.
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from parley import __version__, launcher  # noqa: E402
from parley.update import PLATFORM_KEY, UpdateError, validate_index_files  # noqa: E402

DIST = ROOT / "dist"
WORK = ROOT / "build"
SPEC = ROOT / "packaging" / "parley.spec"
BUILD_PY = ROOT / "src" / "parley" / "_build.py"


class BuildError(Exception):
    pass


def build_py_text(feed_url: str, platform: str = PLATFORM_KEY) -> str:
    return ('"""Written by scripts/build.py into packaged builds only. Never commit."""\n'
            f"FEED_URL = {feed_url!r}\nPLATFORM = {platform!r}\nPUBLIC_KEY = None\n")


def scan_tree(appdir: Path) -> list[tuple[str, str, int]]:
    """(relative posix path, sha256, size) for every file under `appdir` except the index itself, sorted by path."""
    out = []
    for dirpath, _dirs, names in os.walk(appdir):
        for name in names:
            full = Path(dirpath, name)
            rel = full.relative_to(appdir).as_posix()
            if rel == launcher.INDEX_NAME:
                continue
            out.append((rel, launcher.sha256_file(full), full.stat().st_size))
    return sorted(out)


def index_bytes(version: str, entries: list[tuple[str, str, int]]) -> bytes:
    """The `.parley-index.json` / server `files.json` content (same bytes; the launcher and update client read it)."""
    return json.dumps({"schema": 1, "version": version,
                       "files": [{"path": p, "sha256": h, "size": n} for p, h, n in entries]},
                      sort_keys=True).encode()


def bad_paths(entries: list[tuple[str, str, int]]) -> list[str]:
    """Paths the update client would reject (so an update could never install this tree)."""
    bad = []
    for p, h, n in entries:
        try:
            validate_index_files([{"path": p, "sha256": h, "size": n}])
        except UpdateError:
            bad.append(p)
    return bad


def write_index(appdir: Path, version: str, allow_rejected: bool = False) -> str:
    """Scan, validate and write `.parley-index.json`; returns its sha256. Raises BuildError on rejected paths
    (`allow_rejected`: a dev-only escape that builds anyway; such a tree can never be installed by an update)."""
    entries = scan_tree(appdir)
    bad = bad_paths(entries)
    if bad and allow_rejected:
        print(f"WARNING: {len(bad)} path(s) the update client rejects (an update could not install this build): {bad[:5]}",
              file=sys.stderr)
    elif bad:
        raise BuildError(f"{len(bad)} path(s) the update client rejects, e.g. {bad[:5]}")
    try:
        if not bad:
            validate_index_files([{"path": p, "sha256": h, "size": n} for p, h, n in entries])
    except UpdateError as e:  # whole-list rules: duplicates differing in case, size caps, file count
        raise BuildError(f"index not acceptable to the update client: {e}") from e
    data = index_bytes(version, entries)
    (appdir / launcher.INDEX_NAME).write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def layout(frozen_app: Path, frozen_launcher: Path, out: Path, version: str, allow_rejected: bool = False) -> Path:
    """Move the PyInstaller output into the install layout under `out`."""
    if out.exists():
        shutil.rmtree(out)
    appdir = launcher.app_path(out, version)
    appdir.parent.mkdir(parents=True)
    shutil.move(str(frozen_app), str(appdir))
    shutil.move(str(frozen_launcher), str(out / "Parley.exe"))
    digest = write_index(appdir, version, allow_rejected)
    launcher.write_pointer(out, launcher.Pointer(version, index_sha256=digest), backup=False)
    return appdir


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--feed-url", default="", help="update feed base URL baked into the build (default: empty = updates off)")
    ap.add_argument("--allow-rejected-paths", action="store_true",
                    help="dev only: build even if some file names fail the update client's path rule")
    args = ap.parse_args()
    t0 = time.time()
    saved = BUILD_PY.read_text("utf-8") if BUILD_PY.exists() else None
    BUILD_PY.write_text(build_py_text(args.feed_url), "utf-8")
    try:
        for d in (DIST, WORK):
            shutil.rmtree(d, ignore_errors=True)
        subprocess.run([sys.executable, "-m", "PyInstaller", "--noconfirm", "--distpath", str(DIST),
                        "--workpath", str(WORK), str(SPEC)], check=True, cwd=ROOT)
    finally:
        if saved is None:
            BUILD_PY.unlink(missing_ok=True)
        else:
            BUILD_PY.write_text(saved, "utf-8")
    appdir = layout(DIST / "parley-app", DIST / "Parley.exe", DIST / "install", __version__,
                    args.allow_rejected_paths)
    total = sum(p.stat().st_size for p in (DIST / "install").rglob("*") if p.is_file())
    print(f"built parley {__version__} in {time.time() - t0:.0f}s: {DIST / 'install'} ({total / 1e6:.0f} MB), app tree {appdir}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except BuildError as e:
        print(f"build failed: {e}", file=sys.stderr)
        sys.exit(1)
