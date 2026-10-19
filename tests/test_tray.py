"""Tests for the tray wrapper: menu logic with fakes; the real Win32 tray only with a desktop."""

import os
import sys
import threading
import time
import unittest

from parley import tray
from parley.tray import MenuItem, build_menu_model, default_items, dispatch_command


class MenuTests(unittest.TestCase):
    def test_default_menu_layout(self):
        items = default_items("Ready", on_settings=lambda: None)
        labels = [i.label for i in items if not i.separator]
        self.assertEqual(labels, ["Ready", "Settings", "Check for updates", "Open log folder", "Quit"])
        self.assertFalse(items[0].enabled)  # the status line is disabled
        self.assertTrue(all(i.enabled for i in items[1:]))

    def test_version_line_is_a_disabled_item_before_quit(self):
        items = default_items("Ready", version="0.1.0")
        labels = [i.label for i in items if not i.separator]
        self.assertEqual(labels, ["Ready", "Settings", "Check for updates", "Open log folder", "Parley 0.1.0", "Quit"])
        about = next(i for i in items if i.id == "about")
        self.assertFalse(about.enabled)
        self.assertIsNone(about.callback)
        model = build_menu_model(items)
        self.assertFalse(dispatch_command(model, next(c for c, i in model if i.id == "about"), lambda _id: self.fail("ran")))

    def test_check_updates_can_be_hidden(self):
        labels = [i.label for i in default_items("x", can_check_updates=False)]
        self.assertNotIn("Check for updates", labels)

    def test_command_ids_unique_and_dispatch(self):
        calls = []
        items = default_items("s", on_settings=lambda: calls.append("settings"), on_quit=lambda: calls.append("quit"))
        model = build_menu_model(items)
        self.assertEqual(len({c for c, _ in model}), len(model))
        ids = {i.id: c for c, i in model}
        self.assertTrue(dispatch_command(model, ids["settings"], None))
        self.assertTrue(dispatch_command(model, ids["quit"], None))
        self.assertEqual(calls, ["settings", "quit"])

    def test_disabled_and_separator_do_not_run(self):
        seen = []
        model = build_menu_model([MenuItem("a", "A", lambda: seen.append("a"), enabled=False),
                                  MenuItem("sep", "", separator=True)])
        for cid, _ in model:
            self.assertFalse(dispatch_command(model, cid, seen.append))
        self.assertEqual(seen, [])
        self.assertFalse(dispatch_command(model, 99999, None))

    def test_fallback_to_on_menu_command(self):
        seen = []
        model = build_menu_model([MenuItem("open_logs", "Open log folder")])
        self.assertTrue(dispatch_command(model, model[0][0], seen.append))
        self.assertEqual(seen, ["open_logs"])

    def test_callback_error_is_contained(self):
        def boom():
            raise RuntimeError("x")

        model = build_menu_model([MenuItem("a", "A", boom)])
        with self.assertLogs("parley.tray", "ERROR"):
            self.assertTrue(dispatch_command(model, model[0][0], None))

    def test_class_has_api(self):
        for name in ("set_state", "set_tooltip", "balloon", "set_menu", "run", "stop"):
            self.assertTrue(callable(getattr(tray.Tray, name)))


@unittest.skipUnless(sys.platform == "win32" and tray.AVAILABLE, "Windows tray only")
class IconHandleTests(unittest.TestCase):
    def test_make_hicon_every_state(self):
        import win32gui
        from parley import tray_icons

        for state in tray_icons.STATES:
            for f in tray_icons.frames(state, 16):
                h = tray.make_hicon(f)
                self.assertTrue(h)
                win32gui.DestroyIcon(h)


def _has_desktop() -> bool:
    if sys.platform != "win32" or not tray.AVAILABLE:
        return False
    if os.environ.get("CI") or os.environ.get("GITHUB_ACTIONS") or os.environ.get("PARLEY_NO_TRAY_TEST"):
        return False
    return tray.has_notification_area()


@unittest.skipUnless(_has_desktop(), "no interactive desktop / notification area (or CI)")
class RealTrayTests(unittest.TestCase):
    def test_create_modify_delete(self):
        from parley import tray_icons

        t = tray.Tray(tooltip="Parley test")
        th = threading.Thread(target=t.run, daemon=True)
        th.start()
        self.assertTrue(t.wait_ready(5))
        try:
            deadline = time.time() + 3
            while not t.visible and time.time() < deadline:
                time.sleep(0.05)
            self.assertTrue(t.visible, "icon was not added")
            t.set_menu(default_items("test"))
            for state in tray_icons.STATES:
                t.set_state(state)
                t.set_tooltip(f"Parley test {state}")
            time.sleep(0.4)
            self.assertTrue(t.visible)
        finally:
            t.stop()
            th.join(5)
        self.assertFalse(th.is_alive())
        self.assertFalse(t.visible)


@unittest.skipUnless(sys.platform == "win32" and tray.AVAILABLE, "Windows tray")
class DarkMenuTests(unittest.TestCase):
    def test_old_windows_build_does_nothing(self):
        from unittest import mock

        with mock.patch.object(sys, "getwindowsversion", return_value=mock.Mock(build=17763)),                 mock.patch("ctypes.WinDLL") as dll:
            self.assertFalse(tray.apply_menu_theme(0))
        dll.assert_not_called()

    def test_failures_fall_back_silently(self):
        from unittest import mock

        tray._dark_menu_state["warned"] = False
        with mock.patch("ctypes.WinDLL", side_effect=OSError("no uxtheme")):
            self.assertFalse(tray.apply_menu_theme(0))  # no exception
            self.assertFalse(tray.apply_menu_theme(0))
        self.assertTrue(tray._dark_menu_state["warned"])

    def test_missing_ordinal_falls_back(self):
        from unittest import mock

        class NoOrdinals:
            def __getitem__(self, i):
                raise AttributeError("function not found")

        with mock.patch("ctypes.WinDLL", return_value=NoOrdinals()):
            self.assertFalse(tray.apply_menu_theme(0))

    def test_follows_the_theme_read_at_call_time(self):
        from unittest import mock

        from parley import ui

        fn = mock.Mock(return_value=0)
        dll = {135: fn, 136: mock.Mock(), 133: mock.Mock()}
        with mock.patch("ctypes.WinDLL", return_value=dll),                 mock.patch.object(sys, "getwindowsversion", return_value=mock.Mock(build=22631)):
            with mock.patch.object(ui, "system_uses_light_theme", return_value=False):
                self.assertTrue(tray.apply_menu_theme(5))
            fn.assert_called_with(2)  # ForceDark
            with mock.patch.object(ui, "system_uses_light_theme", return_value=True):
                self.assertFalse(tray.apply_menu_theme(5))
            fn.assert_called_with(0)  # Default (light)


if __name__ == "__main__":
    unittest.main()
