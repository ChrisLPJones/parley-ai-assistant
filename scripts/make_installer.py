"""Wrap dist/install (made by scripts/build.py) into a single Inno Setup installer.

    uv run --group build python scripts/build.py
    uv run python scripts/make_installer.py

Result: dist/installer/Parley-Setup-<version>.exe (size and sha256 are printed). The version comes from
`parley.__version__` and is passed to the compiler as /DAppVersion; the script refuses a stale build whose
`current.json` or app dir does not match it. ISCC is looked up on PATH, then in the per-user and the
machine-wide Inno Setup 6 folders. The installer is not signed (see the SignTool note in packaging/parley.iss).
"""
import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from parley import __version__, launcher  # noqa: E402

SOURCE = ROOT / "dist" / "install"
OUTPUT = ROOT / "dist" / "installer"
SCRIPT = ROOT / "packaging" / "parley.iss"
_VERSION = re.compile(r"(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)", re.ASCII)


class InstallerError(Exception):
    pass


def find_iscc(env: dict[str, str] | None = None) -> Path:
    """ISCC.exe from PATH, else the per-user (`%LOCALAPPDATA%\\Programs`) or machine-wide Inno Setup 6 folder."""
    env = os.environ if env is None else env
    on_path = shutil.which("ISCC", path=env.get("PATH"))
    if on_path:
        return Path(on_path)
    for var, sub in (("LOCALAPPDATA", "Programs"), ("ProgramFiles(x86)", ""), ("ProgramFiles", "")):
        base = env.get(var)
        if base:
            cand = Path(base, sub, "Inno Setup 6", "ISCC.exe")
            if cand.is_file():
                return cand
    raise InstallerError("Inno Setup 6 (ISCC.exe) not found: install it (winget install JRSoftware.InnoSetup) or put it on PATH")


def check_source(source: Path, version: str) -> None:
    """The tree must be a complete build of exactly `version` (a stale dist/ would ship the wrong thing)."""
    if not _VERSION.fullmatch(version):
        raise InstallerError(f"bad version {version!r}")
    ptr = launcher.read_pointer(source / launcher.POINTER)
    if ptr is None or ptr.version != version:
        raise InstallerError(f"{source} is not a build of {version} (current.json says {ptr.version if ptr else 'nothing'}): "
                             "run scripts/build.py first")
    for rel in ("Parley.exe", f"app/{version}/{launcher.CHILD_EXE}", f"app/{version}/{launcher.INDEX_NAME}"):
        if not (source / rel).is_file():
            raise InstallerError(f"{source / rel} is missing: run scripts/build.py first")


def iscc_args(iscc: Path, version: str, source: Path, output: Path, script: Path = SCRIPT) -> list[str]:
    return [str(iscc), "/Q", f"/DAppVersion={version}", f"/DSourceDir={source}", f"/DOutputDir={output}", str(script)]


def installer_path(output: Path, version: str) -> Path:
    return output / f"Parley-Setup-{version}.exe"


def make(source: Path = SOURCE, output: Path = OUTPUT, version: str = __version__, iscc: Path | None = None) -> Path:
    check_source(source, version)
    exe = installer_path(output, version)
    exe.unlink(missing_ok=True)
    r = subprocess.run(iscc_args(iscc or find_iscc(), version, source, output), cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0 or not exe.is_file():
        raise InstallerError("ISCC failed:\n" + "\n".join((r.stdout + r.stderr).strip().splitlines()[-15:]))
    return exe


def main() -> int:
    argparse.ArgumentParser(description=__doc__.split("\n")[0]).parse_args()
    exe = make()
    print(f"built {exe}\n  size   {exe.stat().st_size / 1e6:.1f} MB\n  sha256 {launcher.sha256_file(exe)}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except InstallerError as e:
        print(f"installer build failed: {e}", file=sys.stderr)
        sys.exit(1)
