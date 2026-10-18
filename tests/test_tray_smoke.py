"""scripts/tray_smoke.py helpers, and a real run of the whole smoke against `pythonw -m parley --tray` (windowed, so no
console; a throwaway PARLEY_HOME named parley_test_*; Windows only). No models, microphone or network are involved."""
import importlib.util
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("parley_tray_smoke_script", ROOT / "scripts" / "tray_smoke.py")
ts = importlib.util.module_from_spec(_spec)
sys.modules["parley_tray_smoke_script"] = ts
_spec.loader.exec_module(ts)

WIN = sys.platform == "win32"
PYTHONW = Path(sys.executable).with_name("pythonw.exe")


class Pure(unittest.TestCase):
    def test_tree_pids_follows_parents(self):
        table = {1: (0, "a.exe"), 2: (1, "b.exe"), 3: (2, "c.exe"), 4: (9, "other.exe"), 5: (1, "d.exe")}
        self.assertEqual(ts.tree_pids(1, table), {1: "a.exe", 2: "b.exe", 3: "c.exe", 5: "d.exe"})
        self.assertEqual(ts.tree_pids(4, table), {4: "other.exe"})

    def test_only_throwaway_homes_are_allowed(self):
        self.assertTrue(ts.throwaway_ok(Path(tempfile.gettempdir()) / "parley_test_x"))
        self.assertFalse(ts.throwaway_ok(Path(tempfile.gettempdir()) / "parley"))
        real = os.environ.get("LOCALAPPDATA")
        if real:
            self.assertFalse(ts.throwaway_ok(Path(real) / "parley"))

    def test_refuses_the_real_home_before_starting_anything(self):
        if not WIN:
            self.skipTest("Windows only")
        with self.assertRaises(ts.SmokeError):
            ts.run_tray_smoke(Path(sys.executable), Path(tempfile.gettempdir()) / "parley")


@unittest.skipUnless(WIN, "Windows only")
class Real(unittest.TestCase):
    def test_console_detection(self):
        p = subprocess.Popen([sys.executable, "-c", "input()"], stdin=subprocess.PIPE,
                             creationflags=subprocess.CREATE_NEW_CONSOLE)
        try:
            self.assertTrue(ts.has_console(p.pid))
        finally:
            p.kill()
            p.wait()
        w = subprocess.Popen([str(PYTHONW), "-c", "import time; time.sleep(30)"]) if PYTHONW.is_file() else None
        if w is not None:
            try:
                self.assertFalse(ts.has_console(w.pid))
            finally:
                w.kill()
                w.wait()

    @unittest.skipUnless(PYTHONW.is_file(), "pythonw.exe not found")
    def test_whole_smoke_against_a_windowed_python(self):
        with tempfile.TemporaryDirectory(prefix="parley_test_traysmoke_") as home:
            print("PARLEY_HOME =", home)
            done = ts.run_tray_smoke(PYTHONW, Path(home), args=("-m", "parley", "--tray"), expect_exe=None, timeout=120)
            self.assertEqual(len(done), 6, done)  # incl. the informational 'Settings window opened' line


if __name__ == "__main__":
    unittest.main()
