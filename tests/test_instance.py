"""Single-instance mutex: real named mutexes, second process via subprocess, throwaway PARLEY_HOME (Windows only)."""
import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

from parley import instance

WIN = sys.platform == "win32"
HOLDER = ("import sys; from pathlib import Path; from parley import instance\n"
          "h = instance.acquire(Path(sys.argv[1])); print('held' if h else 'none', flush=True)\n"
          "sys.stdin.read()\n")


class Holder:
    """A subprocess that owns the instance mutex for `home` until closed."""

    def __init__(self, home: Path) -> None:
        self.p = subprocess.Popen([sys.executable, "-c", HOLDER, str(home)], stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, text=True)
        assert self.p.stdout.readline().strip() == "held"

    def close(self) -> None:
        self.p.stdin.close()
        self.p.wait(30)


class NameTests(unittest.TestCase):
    def test_name_depends_on_home(self):
        a, b = instance.mutex_name(Path("C:/x/parley_test_a")), instance.mutex_name(Path("C:/x/parley_test_b"))
        self.assertRegex(a, r"^Local\\Parley-[0-9a-f]{8}$")
        self.assertNotEqual(a, b)
        self.assertEqual(a, instance.mutex_name(Path("C:/x/parley_test_a")))


@unittest.skipUnless(WIN, "named mutex is Windows only")
class MutexTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="parley_test_instance_")
        self.home = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def test_second_instance_is_refused_then_allowed_after_exit(self):
        h = Holder(self.home)
        try:
            self.assertIsNone(instance.acquire(self.home))
        finally:
            h.close()
        self.assertIsNotNone(instance.acquire(self.home))

    def test_other_home_does_not_collide(self):
        h = Holder(self.home)
        try:
            with tempfile.TemporaryDirectory(prefix="parley_test_instance_") as other:
                self.assertIsNotNone(instance.acquire(Path(other)))
        finally:
            h.close()

    def test_restarted_waits_for_the_old_instance(self):
        h = Holder(self.home)
        threading.Timer(1.0, h.close).start()
        t0 = time.monotonic()
        self.assertIsNotNone(instance.acquire(self.home, wait_s=15))
        self.assertGreaterEqual(time.monotonic() - t0, 0.5)

    def test_restarted_wait_times_out(self):
        h = Holder(self.home)
        try:
            t0 = time.monotonic()
            self.assertIsNone(instance.acquire(self.home, wait_s=0.6))
            self.assertLess(time.monotonic() - t0, 5)
        finally:
            h.close()

    def test_second_launch_exits_zero_without_loading_anything(self):
        h = Holder(self.home)
        try:
            env = {**os.environ, "PARLEY_HOME": str(self.home)}
            r = subprocess.run([sys.executable, "-m", "parley", "--tray"], env=env, capture_output=True,
                               text=True, timeout=120)
            self.assertEqual(r.returncode, 0, r.stderr[-500:])
            log = (self.home / "logs" / "parley.log").read_text("utf-8")
            self.assertIn("already running", log)
        finally:
            h.close()


if __name__ == "__main__":
    unittest.main()
