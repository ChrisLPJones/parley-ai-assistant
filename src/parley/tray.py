"""Windows notification-area icon over pywin32 (Shell_NotifyIcon, a hidden top-level window, a popup menu).

Thin on purpose: it shows an icon per state (animated by a timer), a tooltip, balloons and a
context menu whose items are plain callables. The window is a normal top-level window (not
message-only) of class `ParleyTrayWindow`, so it receives the TaskbarCreated broadcast (Explorer
restart) and can be found by a second instance. Public methods may be called from any thread:
they queue the work and wake the thread running `run()`. Nothing here raises into the caller.
On other platforms (or without pywin32) `Tray` is a stub with the same API.
"""

from __future__ import annotations

import ctypes
import logging
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from . import tray_icons

log = logging.getLogger(__name__)

WINDOW_CLASS = "ParleyTrayWindow"
OPEN_SETTINGS_MESSAGE = "ParleyOpenSettings"  # registered window message; carries no data

try:  # pywin32 is a Windows-only dependency
    import pywintypes
    import win32api
    import win32con
    import win32event
    import win32gui

    AVAILABLE = True
except ImportError:  # pragma: no cover - exercised on non-Windows
    AVAILABLE = False


@dataclass(frozen=True)
class MenuItem:
    """One context-menu entry. `callback` (if set) runs on click, else `on_menu_command(id)`."""

    id: str
    label: str
    callback: Callable[[], None] | None = None
    enabled: bool = True
    separator: bool = False


def default_items(
    status: str,
    *,
    on_settings: Callable[[], None] | None = None,
    on_check_updates: Callable[[], None] | None = None,
    on_open_logs: Callable[[], None] | None = None,
    on_quit: Callable[[], None] | None = None,
    can_check_updates: bool = True,
) -> list[MenuItem]:
    """The planned menu: disabled status line, Settings, Check for updates, Open log folder, Quit."""
    items = [MenuItem("status", status, enabled=False), MenuItem("sep1", "", separator=True)]
    items.append(MenuItem("settings", "Settings", on_settings))
    if can_check_updates:
        items.append(MenuItem("check_updates", "Check for updates", on_check_updates))
    items.append(MenuItem("open_logs", "Open log folder", on_open_logs))
    items.append(MenuItem("sep2", "", separator=True))
    items.append(MenuItem("quit", "Quit", on_quit))
    return items


_ID_BASE = 1000  # popup-menu command ids start here


def build_menu_model(items: Sequence[MenuItem]) -> list[tuple[int, MenuItem]]:
    """Pure: number the items (command id = _ID_BASE + index). Separators keep an id but are never sent."""
    return [(_ID_BASE + i, item) for i, item in enumerate(items)]


def dispatch_command(
    model: Sequence[tuple[int, MenuItem]], command: int, on_menu_command: Callable[[str], None] | None
) -> bool:
    """Pure-ish: run the item for a popup command id. Returns True when something ran."""
    for cid, item in model:
        if cid != command or item.separator or not item.enabled:
            continue
        try:
            if item.callback is not None:
                item.callback()
            elif on_menu_command is not None:
                on_menu_command(item.id)
            else:
                return False
        except Exception:
            log.exception("Tray menu action %r failed", item.id)
        return True
    return False


if AVAILABLE:
    _WM_TRAY = win32con.WM_USER + 20
    _WM_CALL = win32con.WM_USER + 21
    _TIMER_ID = 1
    _ICON_ID = 1
    _ICON_FLAGS = win32gui.NIF_ICON | win32gui.NIF_MESSAGE | win32gui.NIF_TIP

    # Private DLL instances: setting argtypes/restype on the shared ctypes.windll objects would change them for
    # every other module in the process.
    _user32 = ctypes.WinDLL("user32", use_last_error=True)
    _gdi32 = ctypes.WinDLL("gdi32", use_last_error=True)
    _user32.SetTimer.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint, ctypes.c_void_p]
    _user32.SetTimer.restype = ctypes.c_size_t
    _user32.KillTimer.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    _user32.KillTimer.restype = ctypes.c_int
    _user32.CreateIconIndirect.argtypes = [ctypes.c_void_p]
    _user32.CreateIconIndirect.restype = ctypes.c_void_p
    _gdi32.CreateBitmap.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_uint, ctypes.c_uint, ctypes.c_char_p]
    _gdi32.CreateBitmap.restype = ctypes.c_void_p
    _gdi32.DeleteObject.argtypes = [ctypes.c_void_p]
    _gdi32.DeleteObject.restype = ctypes.c_int

    class _ICONINFO(ctypes.Structure):
        _fields_ = [("fIcon", ctypes.c_int), ("xHotspot", ctypes.c_uint32), ("yHotspot", ctypes.c_uint32),
                    ("hbmMask", ctypes.c_void_p), ("hbmColor", ctypes.c_void_p)]

    def make_hicon(rgba) -> int:
        """HICON from a straight RGBA numpy frame.

        32 bpp BGRA colour bitmap plus an all-zero mask: Windows then uses the per-pixel alpha.
        Colour stays STRAIGHT (not premultiplied): spike showed DrawIconEx premultiplies itself.
        """
        import numpy as np

        h, w, _ = rgba.shape
        bgra = np.empty((h, w, 4), np.uint8)
        bgra[..., 0], bgra[..., 1], bgra[..., 2], bgra[..., 3] = (
            rgba[..., 2], rgba[..., 1], rgba[..., 0], rgba[..., 3])
        # ctypes: win32gui.CreateBitmap only accepts None for the pixel data
        gdi, user = _gdi32, _user32
        color = gdi.CreateBitmap(w, h, 1, 32, bgra.tobytes())
        mask = gdi.CreateBitmap(w, h, 1, 1, bytes(((w + 15) // 16 * 2) * h))
        try:
            info = _ICONINFO(True, 0, 0, mask, color)
            hicon = user.CreateIconIndirect(ctypes.byref(info))
            if not hicon:
                raise ctypes.WinError()
            return hicon
        finally:
            gdi.DeleteObject(color)
            gdi.DeleteObject(mask)

    def _icon_size() -> int:
        n = win32api.GetSystemMetrics(win32con.SM_CXSMICON)
        return n if n in tray_icons.SIZES else min(tray_icons.SIZES, key=lambda s: abs(s - n))

    _dark_menu_state = {"warned": False}

    def apply_menu_theme(hwnd: int) -> bool:
        """Make the next popup menu follow Windows' app theme (dark or light), read NOW so a runtime change is honoured.

        Uses undocumented uxtheme ordinals (135 SetPreferredAppMode / AllowDarkModeForApp, 136 FlushMenuThemes, 133
        AllowDarkModeForWindow), only on Windows 10 build 18362 or later. Any problem (old build, missing ordinal, call
        failure) leaves the normal light menu: logged once at debug level, never raised. Returns True if it was applied."""
        try:
            import sys

            if sys.getwindowsversion().build < 18362:
                return False
            from . import ui

            dark = not ui.system_uses_light_theme()
            uxtheme = ctypes.WinDLL("uxtheme", use_last_error=True)
            set_mode = uxtheme[135]
            set_mode.argtypes, set_mode.restype = [ctypes.c_int], ctypes.c_int
            flush = uxtheme[136]
            flush.argtypes, flush.restype = [], None
            set_mode(2 if dark else 0)  # ForceDark / Default (on build 18362 the argument is a BOOL: same values)
            if dark:
                allow = uxtheme[133]
                allow.argtypes, allow.restype = [ctypes.c_void_p, ctypes.c_int], ctypes.c_int
                allow(hwnd, 1)
            flush()
            return dark
        except Exception as e:  # noqa: BLE001 - cosmetic; the light menu is the fallback
            if not _dark_menu_state["warned"]:
                _dark_menu_state["warned"] = True
                log.debug("Dark popup menu unavailable (%s); using the light menu", e)
            return False

    class Tray:
        def __init__(self, on_menu_command: Callable[[str], None] | None = None, *,
                     on_activate: Callable[[], None] | None = None, tooltip: str = "Parley",
                     on_open_settings: Callable[[], None] | None = None) -> None:
            self.on_menu_command = on_menu_command
            self.on_open_settings = on_open_settings  # a second launch asked for it
            self.on_activate = on_activate  # left click on the icon
            self.visible = False  # the icon is currently in the notification area
            self.created = False  # the window existed: run() returned after a stop/close, not because creation failed
            self._state = "starting"
            self._tooltip = tooltip
            self._items: list[MenuItem] = []
            self._model: list[tuple[int, MenuItem]] = []
            self._queue: list[Callable[[], None]] = []
            self._lock = threading.Lock()
            self._stop = threading.Event()
            self._ready = threading.Event()
            self._hwnd = 0
            self._thread_id = 0
            self._icons: dict[tuple[str, int], int] = {}
            self._frame = 0
            self._taskbar_created = 0
            self._open_settings_msg = 0
            self._class_atom = None
            self._hinst = None

        # ---- public API (any thread) ----
        def set_state(self, state: str) -> None:
            if state not in tray_icons.STATES:
                log.warning("Unknown tray state %r", state)
                return
            self._post(lambda: self._apply_state(state))

        def set_tooltip(self, text: str) -> None:
            self._post(lambda: self._apply_tooltip(text))

        def balloon(self, title: str, text: str) -> None:
            self._post(lambda: self._apply_balloon(title, text))

        def set_menu(self, items: Sequence[MenuItem]) -> None:
            items = list(items)
            self._post(lambda: self._apply_menu(items))

        def run(self) -> None:
            """Create the window and icon on THIS thread and pump messages until stop()."""
            try:
                self._create()
            except Exception:
                log.exception("Could not create the tray window")
                self._cleanup()
                self._ready.set()
                return
            self.created = True
            self._ready.set()
            try:
                while not self._stop.is_set():
                    win32gui.PumpWaitingMessages()
                    # short wait (not infinite) so Ctrl+C reaches a dev console
                    win32event.MsgWaitForMultipleObjects([], False, 50, win32event.QS_ALLINPUT)
            except Exception:
                log.exception("Tray message loop failed")
            finally:
                self._cleanup()

        def stop(self) -> None:
            self._stop.set()
            self._wake()

        def wait_ready(self, timeout: float = 5.0) -> bool:
            return self._ready.wait(timeout)

        # ---- internals (window thread) ----
        def _post(self, fn: Callable[[], None]) -> None:
            with self._lock:
                self._queue.append(fn)
            self._wake()

        def _wake(self) -> None:
            try:
                if self._hwnd:
                    win32gui.PostMessage(self._hwnd, _WM_CALL, 0, 0)
            except Exception:
                pass

        def _drain(self) -> None:
            with self._lock:
                jobs, self._queue = self._queue, []
            for fn in jobs:
                try:
                    fn()
                except Exception:
                    log.exception("Tray update failed")

        def _create(self) -> None:
            self._thread_id = win32api.GetCurrentThreadId()
            self._hinst = win32api.GetModuleHandle(None)
            self._taskbar_created = win32gui.RegisterWindowMessage("TaskbarCreated")
            self._open_settings_msg = win32gui.RegisterWindowMessage(OPEN_SETTINGS_MESSAGE)
            wc = win32gui.WNDCLASS()
            wc.hInstance = self._hinst
            wc.lpszClassName = WINDOW_CLASS
            wc.lpfnWndProc = {
                win32con.WM_DESTROY: self._on_destroy,
                _WM_TRAY: self._on_tray,
                _WM_CALL: self._on_call,
                win32con.WM_TIMER: self._on_timer,
                win32con.WM_CLOSE: self._on_close,
                win32con.WM_QUERYENDSESSION: lambda *a: 1,
                win32con.WM_ENDSESSION: self._on_endsession,
                self._taskbar_created: self._on_taskbar_created,
                self._open_settings_msg: self._on_open_settings,
            }
            try:
                self._class_atom = win32gui.RegisterClass(wc)
            except pywintypes.error:
                # class left from an earlier Tray in this process: reuse it is not possible with a
                # new wndproc, so unregister and retry once
                win32gui.UnregisterClass(WINDOW_CLASS, self._hinst)
                self._class_atom = win32gui.RegisterClass(wc)
            self._hwnd = win32gui.CreateWindow(
                self._class_atom, "Parley", 0, 0, 0, 0, 0, 0, 0, self._hinst, None
            )
            size = _icon_size()
            for state in tray_icons.STATES:
                for i, frame in enumerate(tray_icons.frames(state, size)):
                    self._icons[(state, i)] = make_hicon(frame)
            self._drain()
            self._add_icon()
            self._restart_timer()

        def _cleanup(self) -> None:
            try:
                if self._hwnd:
                    _user32.KillTimer(self._hwnd, _TIMER_ID)
            except Exception:
                pass
            if self.visible:
                try:
                    win32gui.Shell_NotifyIcon(win32gui.NIM_DELETE, (self._hwnd, _ICON_ID))
                except Exception:
                    pass
                self.visible = False
            try:
                if self._hwnd:
                    win32gui.DestroyWindow(self._hwnd)
            except Exception:
                pass
            self._hwnd = 0
            try:
                if self._class_atom:
                    win32gui.UnregisterClass(WINDOW_CLASS, self._hinst)
            except Exception:
                pass
            self._class_atom = None
            for h in self._icons.values():
                try:
                    win32gui.DestroyIcon(h)
                except Exception:
                    pass
            self._icons.clear()

        def _nid(self, flags: int, **extra):
            hicon = self._icons.get((self._state, self._frame % len(tray_icons.STATES[self._state].frames)), 0)
            return (self._hwnd, _ICON_ID, flags, _WM_TRAY, hicon, self._tooltip[:127], *extra.values())

        def _add_icon(self) -> None:
            try:
                win32gui.Shell_NotifyIcon(win32gui.NIM_ADD, self._nid(_ICON_FLAGS))
                self.visible = True
            except Exception:
                self.visible = False
                log.warning("Shell_NotifyIcon(NIM_ADD) failed", exc_info=True)

        def _modify(self, flags: int = _ICON_FLAGS) -> None:
            if not self.visible:
                return
            try:
                win32gui.Shell_NotifyIcon(win32gui.NIM_MODIFY, self._nid(flags))
            except Exception:
                log.warning("Shell_NotifyIcon(NIM_MODIFY) failed", exc_info=True)

        def _restart_timer(self) -> None:
            try:
                _user32.KillTimer(self._hwnd, _TIMER_ID)
            except Exception:
                pass
            spec = tray_icons.STATES[self._state]
            if len(spec.frames) > 1:
                # ctypes: pywin32 has no SetTimer; WM_TIMER arrives in the window procedure
                _user32.SetTimer(self._hwnd, _TIMER_ID, int(1000 / spec.fps), None)

        def _apply_state(self, state: str) -> None:
            if state != self._state:
                self._state, self._frame = state, 0
                self._restart_timer()
            self._modify(win32gui.NIF_ICON)

        def _apply_tooltip(self, text: str) -> None:
            self._tooltip = text
            self._modify(win32gui.NIF_TIP)

        def _apply_balloon(self, title: str, text: str) -> None:
            if not self.visible:
                return
            try:
                win32gui.Shell_NotifyIcon(
                    win32gui.NIM_MODIFY,
                    (self._hwnd, _ICON_ID, win32gui.NIF_INFO, _WM_TRAY, 0, "", text[:255], 10000,
                     title[:63], win32gui.NIIF_INFO),
                )
            except Exception:
                log.warning("Balloon failed", exc_info=True)

        def _apply_menu(self, items: list[MenuItem]) -> None:
            self._items = items
            self._model = build_menu_model(items)

        # ---- window procedure ----
        def _on_call(self, hwnd, msg, wparam, lparam):
            self._drain()
            return 0

        def _on_timer(self, hwnd, msg, wparam, lparam):
            self._frame += 1
            self._modify(win32gui.NIF_ICON)
            return 0

        def _on_taskbar_created(self, hwnd, msg, wparam, lparam):
            self.visible = False
            self._add_icon()
            return 0

        def _on_open_settings(self, hwnd, msg, wparam, lparam):
            if self.on_open_settings:
                try:
                    self.on_open_settings()
                except Exception:
                    log.exception("Open-settings request failed")
            return 0

        def _on_close(self, hwnd, msg, wparam, lparam):
            self._stop.set()
            return 0

        def _on_endsession(self, hwnd, msg, wparam, lparam):
            if wparam:  # TRUE = the session really ends; FALSE = it was cancelled after WM_QUERYENDSESSION
                self._stop.set()
            return 0

        def _on_destroy(self, hwnd, msg, wparam, lparam):
            return 0

        def _on_tray(self, hwnd, msg, wparam, lparam):
            try:
                if lparam == win32con.WM_LBUTTONUP:  # not DBLCLK: a double click already sent an UP
                    if self.on_activate:
                        self.on_activate()
                elif lparam in (win32con.WM_RBUTTONUP, win32con.WM_CONTEXTMENU):
                    self._show_menu()
            except Exception:
                log.exception("Tray click failed")
            return 0

        def _show_menu(self) -> None:
            if not self._model:
                return
            apply_menu_theme(self._hwnd)
            menu = win32gui.CreatePopupMenu()
            try:
                for cid, item in self._model:
                    if item.separator:
                        win32gui.AppendMenu(menu, win32con.MF_SEPARATOR, cid, "")
                    else:
                        flags = win32con.MF_STRING | (0 if item.enabled else win32con.MF_GRAYED | win32con.MF_DISABLED)
                        win32gui.AppendMenu(menu, flags, cid, item.label)
                x, y = win32gui.GetCursorPos()
                try:
                    win32gui.SetForegroundWindow(self._hwnd)  # else the menu will not close on outside click
                except Exception:
                    pass  # Windows refuses it when we are not the foreground process; the menu still shows
                cmd = win32gui.TrackPopupMenu(
                    menu, win32con.TPM_RETURNCMD | win32con.TPM_NONOTIFY | win32con.TPM_RIGHTBUTTON,
                    x, y, 0, self._hwnd, None,
                )
                win32gui.PostMessage(self._hwnd, win32con.WM_NULL, 0, 0)
            finally:
                win32gui.DestroyMenu(menu)
            if cmd:
                dispatch_command(self._model, cmd, self.on_menu_command)

    def post_open_settings() -> bool:
        """Ask the running instance (found by its window class) to open Settings. False if there is none yet."""
        try:
            hwnd = win32gui.FindWindow(WINDOW_CLASS, None)
            if not hwnd:
                return False
            win32gui.PostMessage(hwnd, win32gui.RegisterWindowMessage(OPEN_SETTINGS_MESSAGE), 0, 0)
            return True
        except Exception:
            return False

    def has_notification_area() -> bool:
        """True when an Explorer taskbar with a notification area exists in this session."""
        try:
            return bool(win32gui.FindWindow("Shell_TrayWnd", None))
        except Exception:
            return False

else:

    class Tray:  # type: ignore[no-redef]
        """Stub for platforms without pywin32: same API, does nothing."""

        def __init__(self, on_menu_command=None, *, on_activate=None, tooltip: str = "Parley",
                     on_open_settings=None) -> None:
            self.visible = False
            self.created = True
            self._stop = threading.Event()

        def set_state(self, state: str) -> None: ...
        def set_tooltip(self, text: str) -> None: ...
        def balloon(self, title: str, text: str) -> None: ...
        def set_menu(self, items) -> None: ...

        def run(self) -> None:
            while not self._stop.is_set():
                time.sleep(0.05)

        def stop(self) -> None:
            self._stop.set()

        def wait_ready(self, timeout: float = 5.0) -> bool:
            return True

    def make_hicon(rgba) -> int:
        raise NotImplementedError

    def post_open_settings() -> bool:
        return False

    def has_notification_area() -> bool:
        return False
