"""The one UI thread: owns a withdrawn Tk root and its mainloop. Everything else talks to it through `post(fn)`.

Rules (tkinter is not thread safe):
  * Tk objects are created, used and destroyed only on this thread. Other threads call `post(fn)`; `fn` runs on the UI
    thread within about 50 ms (a queue polled with `after`). Never call a Tk method from anywhere else.
  * The thread starts on the first `post` (or `start()`), is a daemon, and `stop()` ends the mainloop and joins it so Tk
    is torn down on its own thread (else Python can abort at exit with "Tcl_AsyncDelete").
  * DPI: per-system awareness is set before any window; Tk `scaling` follows the system DPI. The sv_ttk theme draws
    fixed-size images and pixel-sized fonts, so fonts are rescaled here (`apply_theme`) and layouts use font-relative
    units (`em`), never raw pixels.
  * Theme follows Windows (`AppsUseLightTheme`, read each time a window opens); the title bar goes dark through
    `DwmSetWindowAttribute(20)`. sv_ttk missing or failing falls back to the `vista` theme (`clam` elsewhere).
"""
import gc
import logging
import queue
import sys
import threading
from collections.abc import Callable

log = logging.getLogger(__name__)

POLL_MS = 50
START_TIMEOUT_S = 10.0
SV_FONTS = ("SunValleyCaptionFont", "SunValleyBodyFont", "SunValleyBodyStrongFont", "SunValleyBodyLargeFont",
            "SunValleySubtitleFont", "SunValleyTitleFont")
cleanup_hooks: list[Callable[[], None]] = []  # run on the UI thread when it ends: drop every reference to Tk objects
_sv_base_sizes: dict[str, int] = {}  # pixel sizes sv_ttk defines (negative in Tk = pixels), read once

# Developer overrides (ui_preview, shot_window): None = follow Windows
theme_override: str | None = None  # "light" | "dark"
scale_override: int | None = None  # percent: 100, 125, 150, 200 ...


# ---------- DPI / theme (pure helpers where possible) ----------

def enable_dpi_awareness() -> bool:
    """Per-system DPI awareness; must run before the first window. Idempotent (an already-set mode is fine)."""
    if sys.platform != "win32":
        return False
    import ctypes

    try:
        hr = ctypes.windll.shcore.SetProcessDpiAwareness(1)  # PROCESS_SYSTEM_DPI_AWARE
        return hr in (0, -2147024891)  # S_OK, E_ACCESSDENIED (already set)
    except (OSError, AttributeError):
        try:
            return bool(ctypes.windll.user32.SetProcessDPIAware())
        except (OSError, AttributeError):
            return False


def system_dpi() -> int:
    if sys.platform != "win32":
        return 96
    import ctypes

    try:
        return int(ctypes.windll.user32.GetDpiForSystem()) or 96
    except (OSError, AttributeError):
        return 96


def system_uses_light_theme() -> bool:
    """True unless Windows apps are set to dark. Not Windows / unreadable = light."""
    if sys.platform != "win32":
        return True
    try:
        import winreg

        with winreg.OpenKey(winreg.HKEY_CURRENT_USER,
                            r"Software\Microsoft\Windows\CurrentVersion\Themes\Personalize") as k:
            return bool(winreg.QueryValueEx(k, "AppsUseLightTheme")[0])
    except OSError:
        return True


def wanted_theme() -> str:
    return theme_override if theme_override in ("light", "dark") else ("light" if system_uses_light_theme() else "dark")


def wanted_scale() -> float:
    """Pixels per 96-dpi pixel: 1.0, 1.5, 2.0 ..."""
    return (scale_override / 100.0) if scale_override else system_dpi() / 96.0


def configure_scaling(root) -> float:
    """Tk `scaling` is pixels per point (72 per inch): 96 dpi = 1.333, 144 dpi = 2.0."""
    factor = wanted_scale()
    root.tk.call("tk", "scaling", factor * 96.0 / 72.0)
    return factor


def apply_theme(root) -> str:
    """Theme (dark or light) and scaled fonts. Returns the theme in use ('dark', 'light' or the fallback's name)."""
    from tkinter import font as tkfont
    from tkinter import ttk

    theme = wanted_theme()
    factor = configure_scaling(root)
    try:
        import sv_ttk

        sv_ttk.set_theme(theme, root)
        root.update_idletasks()  # let the theme's <<ThemeChanged>> handlers run before anyone adds styles
        for name in SV_FONTS:  # sv_ttk sizes are pixels at 96 dpi: scale them ourselves
            f = tkfont.nametofont(name)
            base = _sv_base_sizes.setdefault(name, abs(int(f.cget("size"))))
            f.configure(size=-max(1, round(base * factor)))
        # sv_ttk leaves the default fonts alone: make the classic Tk widgets (Listbox, Text, Menu) match
        family = "Segoe UI Variable Text" if "Segoe UI Variable Text" in tkfont.families() else None
        for name in ("TkDefaultFont", "TkTextFont", "TkMenuFont", "TkHeadingFont"):
            f = tkfont.nametofont(name)
            f.configure(size=-max(1, round(14 * factor)), **({"family": family} if family else {}))
        return theme
    except Exception as e:
        log.info("sv_ttk theme unavailable (%s); using the built-in theme", e)
    style = ttk.Style(root)
    names = style.theme_names()
    style.theme_use("vista" if "vista" in names else "clam")
    return style.theme_use()


def em(widget) -> int:
    """One text line in pixels at the current font and DPI: the unit for padding and window sizes."""
    from tkinter import font as tkfont

    return max(8, tkfont.nametofont("TkDefaultFont").metrics("linespace"))


def set_dark_title_bar(window, dark: bool) -> None:
    """Dark (or light) caption for a Tk toplevel via DWMWA_USE_IMMERSIVE_DARK_MODE. Best effort."""
    if sys.platform != "win32":
        return
    import ctypes

    try:
        window.update_idletasks()
        hwnd = ctypes.windll.user32.GetParent(window.winfo_id()) or window.winfo_id()
        value = ctypes.c_int(1 if dark else 0)
        for attr in (20, 19):  # 20 = Windows 10 20H1+ / 11; 19 = older builds
            if ctypes.windll.dwmapi.DwmSetWindowAttribute(hwnd, attr, ctypes.byref(value), ctypes.sizeof(value)) == 0:
                break
    except Exception as e:  # cosmetic only
        log.debug("Could not set the title bar colour: %s", e)


# ---------- the thread ----------

class UiThread:
    """Owns the Tk root. `root_factory` is injectable so tests can run the queue logic without a display."""

    def __init__(self, root_factory: Callable[[], object] | None = None) -> None:
        self._factory = root_factory or self._tk_root
        self._queue: queue.Queue[Callable[[], None]] = queue.Queue()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()
        self._stopping = threading.Event()
        self.root = None
        self.error: str | None = None  # why Tk could not start

    @staticmethod
    def _tk_root():
        import tkinter

        enable_dpi_awareness()
        root = tkinter.Tk()
        root.withdraw()
        configure_scaling(root)
        return root

    # ---- any thread ----
    def start(self, timeout: float = START_TIMEOUT_S) -> bool:
        """Start the thread if needed; True when the root exists."""
        with self._lock:
            if self._thread is None or not self._thread.is_alive():
                self._ready.clear()
                self._stopping.clear()
                self.error = None
                self._thread = threading.Thread(target=self._run, daemon=True, name="ui")
                self._thread.start()
            thread = self._thread
        if threading.current_thread() is thread:
            return self.root is not None
        self._ready.wait(timeout)
        return self.root is not None

    def post(self, fn: Callable[[], None]) -> bool:
        """Run `fn` on the UI thread soon. Returns False when Tk cannot start (nothing is queued then)."""
        if not self.start():
            return False
        self._queue.put(fn)
        return True

    def call(self, fn: Callable[[], object], timeout: float = 10.0):
        """Run `fn` on the UI thread and return its result (raises its exception). Not for use ON the UI thread."""
        done = threading.Event()
        box: list = []

        def run() -> None:
            try:
                box.append((True, fn()))
            except BaseException as e:  # handed back to the caller
                box.append((False, e))
            done.set()

        if not self.post(run):
            raise RuntimeError(self.error or "the UI could not start")
        if not done.wait(timeout):
            raise TimeoutError("the UI thread did not answer")
        ok, value = box[0]
        if not ok:
            raise value
        return value

    def stop(self, timeout: float = 3.0) -> None:
        """End the mainloop and wait for the thread, so Tk is destroyed where it was created."""
        thread = self._thread
        if thread is None or not thread.is_alive():
            return
        self._stopping.set()
        if threading.current_thread() is not thread:
            thread.join(timeout)

    def alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # ---- the UI thread ----
    def _run(self) -> None:
        try:
            root = self._factory()
        except Exception as e:  # no display, no Tk: the callers get False from post()
            self.error = f"{type(e).__name__}: {e}"[:200]
            log.warning("The settings UI could not start: %s", self.error)
            self._ready.set()
            return
        self.root = root
        try:
            if hasattr(root, "report_callback_exception"):
                root.report_callback_exception = lambda et, ev, tb: log.error("UI callback failed", exc_info=(et, ev, tb))
            root.after(POLL_MS, self._pump)
            self._ready.set()
            root.mainloop()
        finally:
            # Tk variables unset themselves when garbage collected, which must happen on THIS thread, before the root is gone
            for hook in cleanup_hooks:
                try:
                    hook()
                except Exception:
                    log.debug("UI cleanup hook failed", exc_info=True)
            gc.collect()
            self.root = None
            try:
                root.destroy()
            except Exception:
                pass
            gc.collect()

    def _pump(self) -> None:
        while True:
            try:
                fn = self._queue.get_nowait()
            except queue.Empty:
                break
            try:
                fn()
            except Exception:
                log.exception("UI task failed")
        root = self.root
        if root is None:
            return
        if self._stopping.is_set():
            root.quit()
        else:
            root.after(POLL_MS, self._pump)


_ui = UiThread()


def post(fn: Callable[[], None]) -> bool:
    return _ui.post(fn)


def call(fn: Callable[[], object], timeout: float = 10.0):
    return _ui.call(fn, timeout)


def root():
    """The Tk root. UI thread only."""
    return _ui.root


def start() -> bool:
    return _ui.start()


def stop() -> None:
    _ui.stop()


def last_error() -> str | None:
    return _ui.error
