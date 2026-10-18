"""autostart.py against a PRIVATE registry key tree and a test value name: the real `Parley` Run value is never touched.

Run: uv run python -m unittest discover -s tests -v
"""
import sys
import unittest
import uuid
from pathlib import Path

from parley import autostart

WIN = sys.platform == "win32"
if WIN:
    import winreg


@unittest.skipUnless(WIN, "Windows registry")
class AutostartTests(unittest.TestCase):
    NAME = "ParleyTestValue"

    def setUp(self):
        self.base = rf"Software\parley_test_autostart_{uuid.uuid4().hex}"
        self.run, self.approved = self.base + r"\Run", self.base + r"\Approved"
        self.exe = Path(r"C:\Users\Test User\AppData\Local\Programs\Parley\Parley.exe")
        self.real_before = autostart._read(winreg, autostart.RUN_KEY, autostart.VALUE_NAME)
        self.kw = dict(name=self.NAME, exe=self.exe, run_key=self.run, approved_key=self.approved)

    def tearDown(self):
        for sub in (self.run, self.approved, self.base):
            try:
                winreg.DeleteKey(winreg.HKEY_CURRENT_USER, sub)
            except OSError:
                pass
        # the real value must be exactly as before
        self.assertEqual(autostart._read(winreg, autostart.RUN_KEY, autostart.VALUE_NAME), self.real_before)

    def value(self):
        return autostart._read(winreg, self.run, self.NAME)

    def mark(self, first_byte):
        with winreg.CreateKeyEx(winreg.HKEY_CURRENT_USER, self.approved, 0, winreg.KEY_SET_VALUE) as k:
            winreg.SetValueEx(k, self.NAME, 0, winreg.REG_BINARY, bytes([first_byte]) + bytes(11))

    def test_set_and_unset(self):
        self.assertEqual(autostart.status(**self.kw), autostart.State(True, False))
        autostart.set_enabled(True, **self.kw)
        self.assertEqual(self.value(), f'"{self.exe}"')  # quoted path of the launcher
        self.assertEqual(autostart.status(**self.kw), autostart.State(True, True))
        autostart.set_enabled(False, **self.kw)
        self.assertIsNone(self.value())
        self.assertFalse(autostart.status(**self.kw).enabled)
        autostart.set_enabled(False, **self.kw)  # already off: fine

    def test_disabled_in_task_manager_counts_as_disabled(self):
        autostart.set_enabled(True, **self.kw)
        self.mark(0x03)
        st = autostart.status(**self.kw)
        self.assertTrue(st.available)
        self.assertFalse(st.enabled)
        self.assertIn("Task Manager", st.reason)
        self.mark(0x02)  # enabled marker
        self.assertTrue(autostart.status(**self.kw).enabled)

    def test_turning_it_on_clears_the_task_manager_mark(self):
        autostart.set_enabled(True, **self.kw)
        self.mark(0x03)
        autostart.set_enabled(True, **self.kw)
        self.assertIsNone(autostart._read(winreg, self.approved, self.NAME))
        self.assertTrue(autostart.status(**self.kw).enabled)

    def test_a_value_pointing_elsewhere_is_not_ours(self):
        with winreg.CreateKeyEx(winreg.HKEY_CURRENT_USER, self.run, 0, winreg.KEY_SET_VALUE) as k:
            winreg.SetValueEx(k, self.NAME, 0, winreg.REG_SZ, r'"C:\Other\App.exe"')
        st = autostart.status(**self.kw)
        self.assertFalse(st.enabled)
        self.assertIn("replaces", st.reason)

    def test_a_quote_in_the_path_is_refused(self):
        with self.assertRaises(ValueError):
            autostart.command_for(Path('C:\\bad"path\\Parley.exe'))

    def test_a_source_checkout_has_no_launcher(self):
        st = autostart.status(self.NAME, run_key=self.run, approved_key=self.approved)  # exe=None: this install's launcher
        self.assertEqual(st.available, autostart.launcher_path() is not None)
        if not st.available:
            self.assertIn("installed copy", st.reason)
            with self.assertRaises(OSError):
                autostart.set_enabled(True, self.NAME, run_key=self.run, approved_key=self.approved)
            self.assertIsNone(self.value())


if __name__ == "__main__":
    unittest.main()
