"""The push-to-talk hotkey: one pynput listener with a mutable key, shared by the console loop and the tray.

The key can be changed while running (`set_key`, used by Settings later). In capture mode (`capture` is set) the
next key press goes to the capture callback and is passed through to other apps instead of being swallowed, and the
hotkey does nothing. On Windows a low-level filter swallows the hotkey so no other app reacts to it: without that,
Pause in a focused console freezes that console until another key is pressed.
"""
import logging
import threading
from collections.abc import Callable

log = logging.getLogger(__name__)

_KEY_DOWN, _KEY_UP = (0x100, 0x104), (0x101, 0x105)  # WM_KEYDOWN/WM_SYSKEYDOWN, WM_KEYUP/WM_SYSKEYUP


def parse_key(name: str):
    from pynput import keyboard

    name = name.strip().lower()
    if hasattr(keyboard.Key, name):
        return getattr(keyboard.Key, name)
    if len(name) == 1:
        return keyboard.KeyCode.from_char(name)
    raise SystemExit(f"Unknown hotkey {name!r}. Use a pynput key name like pause, f13, scroll_lock, ctrl_r.")


def key_vk(key) -> int | None:
    return getattr(getattr(key, "value", key), "vk", None)


def vk_to_name(vk: int) -> str | None:
    """pynput-style key name for a Windows virtual-key code (what the config file takes), or None if there is none.
    Left/right variants are preferred over the generic ctrl / shift / alt names; letters and digits are one character."""
    from pynput import keyboard

    best = None
    for key in keyboard.Key:
        if key_vk(key) != vk:
            continue
        if best is None or key.name.endswith(("_l", "_r")):
            best = key.name
    if best is None and (0x30 <= vk <= 0x39 or 0x41 <= vk <= 0x5A):
        best = chr(vk).lower()
    return best


class HotkeyListener:
    def __init__(self, app, key_name: str) -> None:
        self.app = app
        self.key_name = key_name
        self.key = parse_key(key_name)
        self.vk = key_vk(self.key)
        # Capture mode: called with the pynput-style virtual-key code (int) of the next key press. None = off.
        self.capture: Callable[[int], None] | None = None
        self.listener = None

    def set_key(self, key_name: str) -> None:
        """Switch the hotkey. Raises SystemExit (like the config loader) for an unknown name; nothing changes then."""
        key = parse_key(key_name)
        self.key, self.vk, self.key_name = key, key_vk(key), key_name

    def start(self) -> None:
        from pynput import keyboard

        self.listener = keyboard.Listener(on_press=self._on_press, on_release=self._on_release,
                                          win32_event_filter=self._win32_filter)
        self.listener.start()
        self.listener.wait()

    def stop(self) -> None:
        if self.listener is not None:
            self.listener.stop()

    def is_alive(self) -> bool:
        return self.listener is not None and self.listener.is_alive()

    # `with HotkeyListener(...) as hl:` mirrors the old `with listener:` of the console loop
    def __enter__(self) -> "HotkeyListener":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()

    def wait(self, stop: threading.Event | None = None) -> None:
        """Block until the listener dies or `stop` is set. Short joins keep Ctrl+C responsive on Windows."""
        while self.is_alive() and not (stop and stop.is_set()):
            self.listener.join(0.5)

    # --- events (pynput thread; keep them quick) ---

    def _on_press(self, k) -> None:
        if self.capture is None and k == self.key:
            self.app.begin_listening()

    def _on_release(self, k) -> None:
        if self.capture is None and k == self.key:
            self.app.end_listening()

    def _win32_filter(self, msg, data) -> bool:
        if self.capture is not None:
            if msg in _KEY_DOWN:
                self._capture(data.vkCode)
            return True  # pass the key through while capturing
        if self.vk is None or data.vkCode != self.vk:
            return True
        if msg in _KEY_DOWN:
            self.app.begin_listening()
        elif msg in _KEY_UP:
            self.app.end_listening()
        self.listener.suppress_event()
        return True  # not reached: suppress_event raises

    def _capture(self, vk: int) -> None:
        cb, self.capture = self.capture, None
        if cb is not None:
            try:
                cb(vk)
            except Exception:
                log.exception("Hotkey capture callback failed")
