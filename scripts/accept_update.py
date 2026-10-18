"""Acceptance run for scripts/release.py: release 0.1.0, 0.1.1 and a deliberately broken 0.1.2 to a LOCAL directory served by
http.server on localhost, then update an installed 0.1.0 with the real update/install/launcher code.

    uv run python scripts/accept_update.py [--ref HEAD] [--keep] [--worktree]

Everything lives under %TEMP%\\parley_test_release (that is also PARLEY_HOME). Never touches %LOCALAPPDATA%\\parley or a real
Programs\\Parley. Takes ~15 minutes (three full builds, one installer run). Why not drive the frozen exe's own updater: the shipped
build only accepts https feeds (`Updater._allow_http_localhost` is a test hook that no config or env can set), so the frozen
`parley-app.exe` cannot talk to a localhost http server. This harness runs the REAL `update.Updater` (with that hook, passed
explicitly), `install.install_version` (which runs the frozen `parley-app.exe --self-check` of the downloaded tree) and the REAL
frozen launcher `Parley.exe` (windowed; it starts `parley-tray.exe`) against the installed tree. A launcher run is the tray smoke test
(scripts/tray_smoke.py: `PARLEY_TRAY_SMOKE=1`, tray window up, second launch, WM_CLOSE, exit 0), which needs no models. Not covered: the app reaching "Ready" and calling confirm() itself
(needs models and a microphone; the harness calls `install.confirm`, the same function) and the MessageBox prompt.
Builds are made with a localhost-http feed URL via release.py's test-only flag; such builds are never shippable.
"""
import argparse
import functools
import http.server
import os
import subprocess
import sys
import tempfile
import threading
import winreg
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import release as rel  # noqa: E402
import tray_smoke  # noqa: E402
from parley import install, launcher, update  # noqa: E402

PORT = 8765
FEED = f"http://127.0.0.1:{PORT}/parley/version.json"
APP_ID_KEY = r"Software\Microsoft\Windows\CurrentVersion\Uninstall\{B4E0FB10-E4F2-4600-9C5B-21DFBFED683D}_is1"


def sabotage(export: Path) -> None:
    """Broken release: passes --self-check and --version (so the installer's check passes) and release.py's own smoke
    runs (which do not set PARLEY_ACCEPT_BREAK), but a start with PARLEY_ACCEPT_BREAK=1 (this script's launcher runs)
    exits 3, so the launcher must roll it back."""
    p = export / "src" / "parley" / "__main__.py"
    text = p.read_text("utf-8")
    marker = "def main() -> None:\n"
    assert marker in text
    p.write_text(text.replace(marker, marker + '    if "--self-check" not in sys.argv and "--version" not in sys.argv \\\n'
                                               '            and os.environ.get("PARLEY_ACCEPT_BREAK") == "1":\n'
                                               '        raise SystemExit(3)\n', 1), "utf-8")


def overlay_worktree(export: Path) -> None:
    """Dev only (--worktree): copy the modified and new (not ignored) files of the working tree over the git export, so
    uncommitted work can be accepted before it is committed. Releases themselves never use the working tree."""
    out = subprocess.run(["git", "-C", str(ROOT), "ls-files", "-m", "-o", "--exclude-standard", "-z"], capture_output=True,
                         check=True).stdout.decode("utf-8").split("\0")
    n = 0
    for rel_path in filter(None, out):
        src = ROOT / rel_path
        if rel_path == "src/parley/__init__.py":
            continue  # release.py rewrites __version__ there (--version-override); never overwrite that
        if src.is_file():
            (export / rel_path).parent.mkdir(parents=True, exist_ok=True)
            (export / rel_path).write_bytes(src.read_bytes())
            n += 1
    print(f"  overlaid {n} working-tree file(s) on the export", flush=True)


def release(version: str, pub: Path, work: Path, hook=None) -> None:
    print(f"\n===== release {version} =====", flush=True)
    w = work / f"build-{version}"
    if ARGS.worktree:
        inner = hook

        def hook(export: Path) -> None:  # noqa: F811 - overlay first, then the sabotage (if any)
            overlay_worktree(export)
            if inner:
                inner(export)
    rc = rel.run(["--ref", ARGS.ref, "--notes", f"Acceptance build {version}", "--feed-url", FEED, "--test-http-feed",
                  "--target", str(pub / "parley"), "--version-override", version, "--work-dir", str(w),
                  "--smoke-home", str(work)], export_hook=hook)
    assert rc == 0
    install._rmtree(w)  # the export + its venv are large; the artifacts are in the publish dir now


def launcher_run(root: Path, env: dict) -> int:
    """Run the installed launcher through the tray smoke test; 0 = the started version came up and quit cleanly."""
    try:
        for line in tray_smoke.run_tray_smoke(root / "Parley.exe", Path(env["PARLEY_HOME"]), env=env, timeout=300):
            print("   ", line, flush=True)
    except tray_smoke.SmokeError as e:
        print(f"  Parley.exe tray smoke FAILED: {e}", flush=True)
        return 1
    print("  Parley.exe tray smoke -> exit 0", flush=True)
    return 0


def pointer(root: Path) -> str:
    p = launcher.read_pointer(root / launcher.POINTER)
    return f"version={p.version} previous={p.previous} state={p.state} skip={p.skip}"


def main() -> int:
    global ARGS
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ref", default="HEAD")
    ap.add_argument("--keep", action="store_true", help="keep the work dir")
    ap.add_argument("--worktree", action="store_true",
                    help="dev only: build the working tree's uncommitted changes (overlay on the --ref export)")
    ARGS = ap.parse_args()
    work = Path(tempfile.gettempdir()) / "parley_test_release"
    assert rel.smoke_home_ok(work)
    print(f"PARLEY_HOME={work}", flush=True)
    if work.exists():
        install._rmtree(work)
    work.mkdir(parents=True)
    env = {**rel.clean_env(), "PARLEY_HOME": str(work), "PARLEY_ACCEPT_BREAK": "1"}  # the latter only matters to the sabotaged build
    try:  # the Setup shares one AppId with a real install: never run it over a real one
        winreg.OpenKey(winreg.HKEY_CURRENT_USER, APP_ID_KEY).Close()
        raise SystemExit("Parley is registered as installed for this user: refusing to run the Setup (uninstall it first)")
    except OSError:
        pass
    pub = work / "pub"
    (pub / "parley").mkdir(parents=True)
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(pub))
    handler.log_message = lambda *a, **k: None  # type: ignore[attr-defined]
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", PORT), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    inst_dir = work / "Programs" / "Parley"
    try:
        release("0.1.0", pub, work)
        # no http in the shipped client:
        try:
            update.Updater(FEED, "0.1.0")
            raise SystemExit("FAIL: the shipped Updater accepted an http feed")
        except update.UpdateError as e:
            print(f"shipped Updater rejects http as designed: {e}")
        print("\n===== install 0.1.0 with the real Setup (silent, throwaway dir) =====", flush=True)
        setup = pub / "parley" / "Parley-Setup-0.1.0.exe"
        log = work / "setup.log"
        subprocess.run([str(setup), "/VERYSILENT", "/SUPPRESSMSGBOXES", "/NORESTART", "/SP-", f"/DIR={inst_dir}", f"/LOG={log}"],
                       check=True, env={**os.environ, "MSYS_NO_PATHCONV": "1"}, timeout=900)
        print(" ", pointer(inst_dir))

        release("0.1.1", pub, work)
        print("\n===== update 0.1.0 -> 0.1.1 =====", flush=True)
        u = update.Updater(FEED, "0.1.0", update.PLATFORM_KEY, inst_dir / "update_state.json", _allow_http_localhost=True)
        m = u.check()
        assert m is not None and m.version == "0.1.1", m
        files = u.fetch_index(m)
        fetched: list[int] = []
        orig = u.download_blob
        u.download_blob = lambda f, dest: (fetched.append(f.size), orig(f, dest))[1]  # type: ignore[method-assign]
        install.install_version(inst_dir, u, m, files)
        new = launcher.app_path(inst_dir, "0.1.1")
        linked = sum(1 for f in files if (new / f.path).stat().st_nlink > 1)
        print(f"  {len(files)} files: {len(fetched)} downloaded ({sum(fetched) / 1e6:.1f} MB), {linked} hardlinked from 0.1.0")
        print(" ", pointer(inst_dir))
        assert launcher_run(inst_dir, env) == 0
        assert install.confirm(inst_dir, "0.1.1")
        print(" ", pointer(inst_dir), "(confirmed)")

        release("0.1.2", pub, work, hook=sabotage)
        print("\n===== update 0.1.1 -> broken 0.1.2 =====", flush=True)
        u = update.Updater(FEED, "0.1.1", update.PLATFORM_KEY, inst_dir / "update_state.json", _allow_http_localhost=True)
        m = u.check()
        assert m is not None and m.version == "0.1.2"
        install.install_version(inst_dir, u, m, u.fetch_index(m))
        print(" ", pointer(inst_dir), "(self-check of the downloaded tree passed)")
        rc = launcher_run(inst_dir, env)  # 0.1.2 exits 3 while pending -> rolled back, 0.1.1 started
        print(" ", pointer(inst_dir))
        ptr = launcher.read_pointer(inst_dir / launcher.POINTER)
        assert rc == 0 and ptr.version == "0.1.1" and ptr.skip == ["0.1.2"] and ptr.state == "good", (rc, ptr)
        assert install.is_skipped(inst_dir, "0.1.2")
        try:
            install.install_version(inst_dir, u, m, u.fetch_index(m))
            raise SystemExit("FAIL: the skipped version was installed again")
        except install.InstallError as e:
            print(f"  not offered again: {e}")
        print("\nACCEPTANCE PASSED")
        return 0
    finally:
        srv.shutdown()
        un = inst_dir / "unins000.exe"
        if un.is_file():
            subprocess.run([str(un), "/VERYSILENT", "/SUPPRESSMSGBOXES", "/NORESTART"], env={**os.environ, "MSYS_NO_PATHCONV": "1"},
                           timeout=300)
        if not ARGS.keep:
            install._rmtree(work)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except rel.ReleaseError as e:
        print(f"FAILED: {e}", file=sys.stderr)
        sys.exit(1)
