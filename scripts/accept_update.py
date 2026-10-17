"""Acceptance run for scripts/release.py: release 0.1.0, 0.1.1 and a deliberately broken 0.1.2 to a LOCAL directory served by
http.server on localhost, then update an installed 0.1.0 with the real update/install/launcher code.

    uv run python scripts/accept_update.py [--ref HEAD] [--keep]

Everything lives under %TEMP%\\parley_test_release (that is also PARLEY_HOME). Never touches %LOCALAPPDATA%\\parley or a real
Programs\\Parley. Takes ~15 minutes (three full builds, one installer run). Why not drive the frozen exe's own updater: the shipped
build only accepts https feeds (`Updater._allow_http_localhost` is a test hook that no config or env can set), so the frozen
`parley-app.exe` cannot talk to a localhost http server. This harness runs the REAL `update.Updater` (with that hook, passed
explicitly), `install.install_version` (which runs the frozen `parley-app.exe --self-check` of the downloaded tree) and the REAL
frozen launcher `Parley.exe` against the installed tree. Not covered: the app reaching "Ready" and calling confirm() itself
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
from parley import install, launcher, update  # noqa: E402

PORT = 8765
FEED = f"http://127.0.0.1:{PORT}/parley/version.json"
APP_ID_KEY = r"Software\Microsoft\Windows\CurrentVersion\Uninstall\{B4E0FB10-E4F2-4600-9C5B-21DFBFED683D}_is1"


def sabotage(export: Path) -> None:
    """Broken release: passes --self-check and --version (so release.py's smoke and the installer's check pass) but any
    normal start exits 3, so the launcher must roll it back."""
    p = export / "src" / "parley" / "__main__.py"
    text = p.read_text("utf-8")
    marker = "def main() -> None:\n"
    assert marker in text
    p.write_text(text.replace(marker, marker + '    if "--self-check" not in sys.argv and "--version" not in sys.argv:\n'
                                               '        raise SystemExit(3)\n', 1), "utf-8")


def release(version: str, pub: Path, work: Path, hook=None) -> None:
    print(f"\n===== release {version} =====", flush=True)
    w = work / f"build-{version}"
    rc = rel.run(["--ref", ARGS.ref, "--notes", f"Acceptance build {version}", "--feed-url", FEED, "--test-http-feed",
                  "--target", str(pub / "parley"), "--version-override", version, "--work-dir", str(w),
                  "--smoke-home", str(work)], export_hook=hook)
    assert rc == 0
    install._rmtree(w)  # the export + its venv are large; the artifacts are in the publish dir now


def launcher_run(root: Path, env: dict) -> int:
    r = subprocess.run([str(root / "Parley.exe"), "--list-devices"], env=env, capture_output=True, text=True, timeout=300)
    print(f"  Parley.exe --list-devices -> exit {r.returncode}", flush=True)
    return r.returncode


def pointer(root: Path) -> str:
    p = launcher.read_pointer(root / launcher.POINTER)
    return f"version={p.version} previous={p.previous} state={p.state} skip={p.skip}"


def main() -> int:
    global ARGS
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ref", default="HEAD")
    ap.add_argument("--keep", action="store_true", help="keep the work dir")
    ARGS = ap.parse_args()
    work = Path(tempfile.gettempdir()) / "parley_test_release"
    assert rel.smoke_home_ok(work)
    print(f"PARLEY_HOME={work}", flush=True)
    if work.exists():
        install._rmtree(work)
    work.mkdir(parents=True)
    env = {**rel.clean_env(), "PARLEY_HOME": str(work)}
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
