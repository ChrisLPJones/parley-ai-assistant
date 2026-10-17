"""HotkeyListener: key switching, capture mode, the Win32 swallow filter (fakes; no real keyboard hook)."""
import unittest
from types import SimpleNamespace

from parley.hotkey import HotkeyListener

WM_KEYDOWN, WM_KEYUP = 0x100, 0x101
VK_PAUSE, VK_A = 0x13, 0x41


class FakeApp:
    def __init__(self) -> None:
        self.events: list[str] = []

    def begin_listening(self): self.events.append("begin")
    def end_listening(self): self.events.append("end")


class Suppressed(Exception):
    pass


class FakePynput:
    def suppress_event(self):
        raise Suppressed  # like pynput: aborts the filter and drops the event


class HotkeyTests(unittest.TestCase):
    def make(self, key="pause"):
        app = FakeApp()
        hl = HotkeyListener(app, key)
        hl.listener = FakePynput()
        return app, hl

    def press(self, hl, vk, msg=WM_KEYDOWN):
        try:
            return hl._win32_filter(msg, SimpleNamespace(vkCode=vk))
        except Suppressed:
            return "suppressed"

    def test_hotkey_is_swallowed_and_drives_the_app(self):
        app, hl = self.make()
        self.assertEqual(self.press(hl, VK_PAUSE), "suppressed")
        self.assertEqual(self.press(hl, VK_PAUSE, WM_KEYUP), "suppressed")
        self.assertEqual(app.events, ["begin", "end"])

    def test_other_keys_pass_through(self):
        app, hl = self.make()
        self.assertTrue(self.press(hl, VK_A))
        self.assertEqual(app.events, [])

    def test_set_key_changes_the_vk(self):
        app, hl = self.make()
        hl.set_key("f13")
        self.assertTrue(self.press(hl, VK_PAUSE))  # no longer the hotkey
        self.assertEqual(self.press(hl, 0x7C), "suppressed")  # F13
        self.assertEqual(app.events, ["begin"])

    def test_unknown_key_raises_and_changes_nothing(self):
        _, hl = self.make()
        with self.assertRaises(SystemExit):
            hl.set_key("nonsense")
        self.assertEqual((hl.key_name, hl.vk), ("pause", VK_PAUSE))

    def test_capture_mode_passes_the_key_through_once(self):
        app, hl = self.make()
        got = []
        hl.capture = got.append
        self.assertTrue(self.press(hl, VK_PAUSE))  # even the current hotkey is not swallowed
        self.assertEqual(got, [VK_PAUSE])
        self.assertIsNone(hl.capture)
        self.assertEqual(app.events, [])
        self.assertEqual(self.press(hl, VK_PAUSE), "suppressed")  # normal again

    def test_pynput_press_ignored_while_capturing(self):
        app, hl = self.make()
        hl.capture = lambda vk: None
        hl._on_press(hl.key)
        self.assertEqual(app.events, [])


if __name__ == "__main__":
    unittest.main()
