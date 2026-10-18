"""Smoke test of a packaged tray exe (`parley-tray.exe`, or the launcher `Parley.exe` that starts it). Windows only.

    uv run python scripts/tray_smoke.py --exe dist/install/Parley.exe [--home <temp>/parley_test_tray]

Starts the exe with `PARLEY_TRAY_SMOKE=1` and a throwaway `PARLEY_HOME` (its folder must be named `parley_test_*`; the
real Parley home is refused). In that mode (src/parley/tray_mode.py `smoke_requested`) the app creates its tray window and
does nothing else: no config, no models, no downloads, no microphone. Checks, in order:
  1. a window of class `ParleyTrayWindow` owned by a process of the started tree appears, owned by `parley-tray.exe`;
  2. no process of the tree has a console (the exe is windowed);
  3. the single-instance mutex for this home exists;
  4. a second launch exits 0 at once and asks the first one to open Settings (log line), leaving the first running;
  5. WM_CLOSE to the window ends the tree with exit code 0, and no process, window or mutex is left.
The window is found by process id, never by class alone, so a Parley you are running yourself is never touched.
"""
import argparse
import ctypes
import os
import subprocess
import sys
import tempfile
import time
from ctypes import wintypes
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

WINDOW_CLASS = "ParleyTrayWindow"  # parley.tray.WINDOW_CLASS (not imported: that needs pywin32 at import time)
TRAY_EXE = "parley-tray.exe"
TH32CS_SNAPPROCESS = 0x2
WM_CLOSE = 0x10
DETACHED_PROCESS = 0x8
CREATE_NO_WINDOW = 0x08000000


class SmokeError(Exception):
    pass


class _ProcessEntry(ctypes.Structure):
    _fields_ = [("dwSize", wintypes.DWORD), ("cntUsage", wintypes.DWORD), ("th32ProcessID", wintypes.DWORD),
                ("th32DefaultHeapID", ctypes.c_size_t), ("th32ModuleID", wintypes.DWORD), ("cntThreads", wintypes.DWORD),
                ("th32ParentProcessID", wintypes.DWORD), ("pcPriClassBase", wintypes.LONG), ("dwFlags", wintypes.DWORD),
                ("szExeFile", ctypes.c_wchar * 260)]


def process_table() -> dict[int, tuple[int, str]]:
    """pid -> (parent pid, exe name) of every running process."""
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    k32.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
    k32.Process32FirstW.argtypes = k32.Process32NextW.argtypes = [wintypes.HANDLE, ctypes.POINTER(_ProcessEntry)]
    k32.CloseHandle.argtypes = [wintypes.HANDLE]
    snap = k32.CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0)
    out: dict[int, tuple[int, str]] = {}
    try:
        entry = _ProcessEntry()
        entry.dwSize = ctypes.sizeof(_ProcessEntry)
        ok = k32.Process32FirstW(snap, ctypes.byref(entry))
        while ok:
            out[entry.th32ProcessID] = (entry.th32ParentProcessID, entry.szExeFile)
            ok = k32.Process32NextW(snap, ctypes.byref(entry))
    finally:
        k32.CloseHandle(snap)
    return out


def tree_pids(root_pid: int, table: dict[int, tuple[int, str]] | None = None) -> dict[int, str]:
    """pid -> exe name for `root_pid` and all its descendants."""
    table = table if table is not None else process_table()
    found = {root_pid: table.get(root_pid, (0, "?"))[1]}
    grew = True
    while grew:
        grew = False
        for pid, (ppid, name) in table.items():
            if ppid in found and pid not in found:
                found[pid] = name
                grew = True
    return found


def windows_of(pids: set[int], cls: str = WINDOW_CLASS) -> list[int]:
    """Top-level windows of class `cls` owned by one of `pids`."""
    import win32gui
    import win32process

    hits: list[int] = []

    def visit(hwnd, _):
        if win32gui.GetClassName(hwnd) == cls and win32process.GetWindowThreadProcessId(hwnd)[1] in pids:
            hits.append(hwnd)
        return True

    win32gui.EnumWindows(visit, None)
    return hits


_HAS_CONSOLE = (
    "import ctypes,sys\n"
    "k=ctypes.WinDLL('kernel32',use_last_error=True)\n"
    "k.FreeConsole()\n"
    "sys.exit(0 if k.AttachConsole(int(sys.argv[1])) else 1)\n"
)


def has_console(pid: int) -> bool:
    """True if process `pid` is attached to a console (AttachConsole from a helper that has none itself)."""
    r = subprocess.run([sys.executable, "-c", _HAS_CONSOLE, str(pid)], creationflags=DETACHED_PROCESS,
                       capture_output=True, timeout=30)
    return r.returncode == 0


def mutex_exists(name: str) -> bool:
    import pywintypes
    import win32api
    import win32event

    try:
        h = win32event.OpenMutex(0x00100000, False, name)  # SYNCHRONIZE
    except pywintypes.error:
        return False
    win32api.CloseHandle(h)
    return True


def throwaway_ok(home: Path) -> bool:
    real = os.environ.get("LOCALAPPDATA")
    if real and home.resolve() == (Path(real) / "parley").resolve():
        return False
    return home.name.startswith("parley_test_")


def _wait(cond, timeout: float, what: str, proc: subprocess.Popen | None = None):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        v = cond()
        if v:
            return v
        if proc is not None and proc.poll() is not None:
            raise SmokeError(f"{what}: the process ended first (exit {proc.returncode})")
        time.sleep(0.25)
    raise SmokeError(f"timed out after {timeout:.0f}s waiting for {what}")


def run_tray_smoke(exe: Path, home: Path, *, timeout: float = 90.0, env: dict[str, str] | None = None,
                   args: tuple[str, ...] = (), expect_exe: str | None = TRAY_EXE) -> list[str]:
    """Run the checks in the module docstring against `exe` (+ `args`; tests use pythonw.exe -m parley --tray with
    `expect_exe=None`). Returns a list of what was verified; raises SmokeError."""
    import win32gui

    from parley import instance

    if not throwaway_ok(home):
        raise SmokeError(f"smoke home {home} must be a throwaway dir named parley_test_<topic>")
    if not exe.is_file():
        raise SmokeError(f"{exe} not found")
    home.mkdir(parents=True, exist_ok=True)
    env = {**(env if env is not None else os.environ), "PARLEY_HOME": str(home), "PARLEY_TRAY_SMOKE": "1"}
    done: list[str] = []
    proc = subprocess.Popen([str(exe), *args], env=env, cwd=exe.parent, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL)
    seen: dict[int, str] = {}
    try:
        def find():
            pids = tree_pids(proc.pid)
            seen.update(pids)
            return windows_of(set(pids))

        hwnd = _wait(find, timeout, f"the {WINDOW_CLASS} window", proc)[0]
        seen.update(tree_pids(proc.pid))
        import win32process

        owner = win32process.GetWindowThreadProcessId(hwnd)[1]
        if expect_exe and seen.get(owner, "?").lower() != expect_exe:
            raise SmokeError(f"the tray window belongs to {seen.get(owner)!r}, expected {expect_exe}")
        done.append(f"window {WINDOW_CLASS} owned by {seen.get(owner)} (pid {owner}); processes {sorted(set(seen.values()))}")

        consoles = [f"{n} ({p})" for p, n in seen.items() if has_console(p)]
        if consoles:
            raise SmokeError(f"process(es) with a console: {consoles}")
        done.append(f"no console on any of {len(seen)} process(es)")

        mutex = instance.mutex_name(home)
        if not mutex_exists(mutex):
            raise SmokeError(f"mutex {mutex} does not exist")
        done.append(f"mutex {mutex} exists")

        second = subprocess.run([str(exe), *args], env=env, cwd=exe.parent, stdin=subprocess.DEVNULL, capture_output=True,
                                timeout=60)
        if second.returncode != 0:
            raise SmokeError(f"second launch exited {second.returncode}, expected 0")
        log = home / "logs" / "parley.log"
        _wait(lambda: log.is_file() and "Settings requested" in log.read_text("utf-8", "replace"), 30,
              "the 'Settings requested' log line from the second launch")
        if proc.poll() is not None or not win32gui.IsWindow(hwnd):
            raise SmokeError("the first instance did not survive the second launch")
        done.append("second launch exited 0 and asked the first instance to open Settings")
        # The Settings window itself (needs a desktop for Tk): reported, not required, so a runner without one still passes.
        # A build that lost tkinter fails --self-check instead.
        def settings_outcome():
            text = log.read_text("utf-8", "replace")
            if "Settings window opened" in text:
                return "opened"
            return "no display" if "settings UI could not start" in text else None

        try:
            outcome = _wait(settings_outcome, 20, "the Settings window", proc)
        except SmokeError:
            outcome = "no answer"
        done.append("Settings window opened" if outcome == "opened" else f"Settings window not opened ({outcome}); not required")

        seen.update(tree_pids(proc.pid))
        win32gui.PostMessage(hwnd, WM_CLOSE, 0, 0)
        try:
            code = proc.wait(30)
        except subprocess.TimeoutExpired:
            raise SmokeError("the process did not exit within 30 s of WM_CLOSE") from None
        if code != 0:
            raise SmokeError(f"exit code {code} after WM_CLOSE, expected 0")
        time.sleep(0.5)
        table = process_table()
        left = [f"{n} ({p})" for p, n in seen.items() if p in table and table[p][1] == n]
        if left:
            raise SmokeError(f"processes left behind: {left}")
        if windows_of(set(seen)) or mutex_exists(mutex):
            raise SmokeError("window or mutex left behind")
        done.append("WM_CLOSE -> exit 0, no process, window or mutex left")
        return done
    finally:
        if proc.poll() is None:
            subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)], capture_output=True, timeout=30,
                           creationflags=CREATE_NO_WINDOW)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--exe", required=True, type=Path, help="parley-tray.exe or the launcher Parley.exe")
    ap.add_argument("--home", type=Path, help="throwaway PARLEY_HOME named parley_test_* (default: a new temp dir)")
    args = ap.parse_args()
    home = args.home or Path(tempfile.mkdtemp(prefix="parley_test_traysmoke_"))
    print(f"PARLEY_HOME={home}", flush=True)
    try:
        for line in run_tray_smoke(args.exe, home):
            print("  ok:", line)
    except SmokeError as e:
        print(f"FAILED: {e}", file=sys.stderr)
        return 1
    print("TRAY SMOKE PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
