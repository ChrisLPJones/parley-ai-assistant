"""Shared tray / Advanced page actions with fakes: no real Explorer window, no network.

Run: uv run python -m unittest discover -s tests -v
Scratch dirs are temp dirs named parley_test_actions_*; the real Parley home is never touched.
"""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from parley import actions, logsetup


class FolderTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="parley_test_actions_")
        self.addCleanup(self._tmp.cleanup)
        self.home = Path(self._tmp.name)

    def test_log_folder_is_created_and_opened(self):
        opened = []
        self.assertEqual(actions.open_log_folder(self.home, opened.append), "")
        self.assertEqual(opened, [str(logsetup.log_dir(self.home))])
        self.assertTrue(logsetup.log_dir(self.home).is_dir())

    def test_parley_folder_is_the_home(self):
        opened = []
        self.assertEqual(actions.open_parley_folder(self.home, opened.append), "")
        self.assertEqual(opened, [str(self.home)])

    def test_failure_is_a_sentence_not_an_exception(self):
        def boom(_p):
            raise OSError("denied")

        self.assertEqual(actions.open_log_folder(self.home, boom), "The folder could not be opened.")

    def test_no_startfile_says_windows_is_needed(self):
        from unittest import mock

        with mock.patch.object(actions.os, "startfile", None, create=True):
            self.assertIn("Windows", actions.open_parley_folder(self.home))


class UpdateTests(unittest.TestCase):
    def test_without_a_flow_it_says_why(self):
        said = []
        actions.check_for_updates(None, said.append)
        self.assertEqual(said, [actions.NO_UPDATES])

    def test_with_a_flow_it_uses_check_now(self):
        said = []
        flow = SimpleNamespace(check_now=lambda notify: notify("Parley 0.1.0 is up to date."))
        actions.check_for_updates(flow, said.append)
        self.assertEqual(said, ["Parley 0.1.0 is up to date."])


if __name__ == "__main__":
    unittest.main()
