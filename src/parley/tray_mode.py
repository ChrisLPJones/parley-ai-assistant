"""`parley --tray`: Parley as a notification-area app (Windows).

Threads: the MAIN thread runs the tray message loop; a boot thread loads the config and builds `Parley` (the tray shows
"starting" at once); a poll thread turns `Parley.activity()` into the icon state; pynput runs the hotkey on its own
thread. Menu callbacks are handed to worker threads so a slow one never freezes the message pump.
Startup problems (a bad setting, a failed download) show the error icon plus a balloon and the process stays alive; an
unexpected crash exits non-zero so the launcher's rollback still works. Plain `parley` stays the console loop.
"""
import faulthandler
import logging
import os
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path

from . import config, credstore, download, instance, logsetup, tray, ui

log = logging.getLogger(__name__)

TRAY_EXE_STEM = "parley-tray"  # the windowed exe (packaging/parley.spec): its name implies --tray
SMOKE_ENV = "PARLEY_TRAY_SMOKE"
SMOKE_HOME_PREFIX = "parley_test_"
SMOKE_TIMEOUT_S = 300.0
MAX_ERROR_CHARS = 200
DEBOUNCE_S = 0.15
POLL_S = 0.1
SETTINGS_DEBOUNCE_S = 1.5  # one click (or double click) = one request; the window itself is idempotent anyway
WELCOME_MARK = "tray_welcomed"
WELCOME_MAX_S = 120.0  # give up waiting for the spoken welcome after this long
IDLE_POLL_S = 0.5  # how often the welcome and the GPU restart look at whether Parley is idle
IDLE_CHECKS = 3  # consecutive idle looks before the automatic restart
_faulthandler_file = None  # kept open for the life of the process


def implied() -> bool:
    """True when this process is the windowed tray exe (no `--tray` flag needed)."""
    return bool(getattr(sys, "frozen", False)) and Path(sys.executable).stem.lower() == TRAY_EXE_STEM


def smoke_requested(env: dict | None = None) -> bool:
    """Test-only mode for the packaged-build smoke test (scripts/tray_smoke.py): create the tray window, do NOT build
    Parley or download anything, and exit 0 on WM_CLOSE. Gated twice so production never enters it: the variable
    `PARLEY_TRAY_SMOKE=1` AND a `PARLEY_HOME` whose folder name starts with `parley_test_` (never the real home)."""
    env = os.environ if env is None else env
    home = env.get("PARLEY_HOME")
    return env.get(SMOKE_ENV) == "1" and bool(home) and Path(home).name.startswith(SMOKE_HOME_PREFIX)


def short_error(msg: object) -> str:
    """Text safe to show in a balloon: redacted, one line, at most 200 characters."""
    text = credstore.redact(" ".join(str(msg).split()))
    return text if len(text) <= MAX_ERROR_CHARS else text[:MAX_ERROR_CHARS - 3] + "..."


def threaded(fn: Callable[[], None], name: str = "menu") -> Callable[[], None]:
    """Wrap a menu callback so it runs on a worker thread: the message pump returns at once."""
    def run() -> None:
        def work() -> None:
            try:
                fn()
            except Exception:
                log.exception("Tray action %r failed", name)
        threading.Thread(target=work, daemon=True, name=f"tray-{name}").start()
    return run


# ---------- crashes ----------

class CrashHooks:
    """Log unhandled exceptions (redacted by the log handlers) and show ONE generic balloon per session.
    The balloon never carries exception text: it could contain anything."""

    def __init__(self, notify: Callable[[str, str], None]) -> None:
        self._notify, self._shown, self._lock = notify, False, threading.Lock()

    def install(self) -> None:
        sys.excepthook = self.on_exception
        threading.excepthook = self.on_thread_exception

    def on_exception(self, exc_type, exc, tb) -> None:
        log.critical("Unhandled exception", exc_info=(exc_type, exc, tb))
        self._balloon()

    def on_thread_exception(self, args) -> None:
        if args.exc_type is SystemExit:
            return
        name = args.thread.name if args.thread else "?"
        log.critical("Unhandled exception in thread %s", name, exc_info=(args.exc_type, args.exc_value, args.exc_traceback))
        self._balloon()

    def _balloon(self) -> None:
        with self._lock:
            if self._shown:
                return
            self._shown = True
        try:
            self._notify("Parley hit a problem", "Something went wrong. Open the log folder from the tray menu for details.")
        except Exception:
            pass


def enable_faulthandler(home: Path) -> None:
    """Native crashes write their stack to `<home>/logs/crash.log` (frames only, no values; a separate file because
    the rotating handler may rename parley.log while faulthandler holds it open)."""
    global _faulthandler_file
    try:
        d = logsetup.log_dir(home)
        d.mkdir(parents=True, exist_ok=True)
        _faulthandler_file = open(d / "crash.log", "a", encoding="utf-8")
        faulthandler.enable(file=_faulthandler_file, all_threads=True)
    except OSError as e:
        log.warning("faulthandler is off: %s", e)


# ---------- icon state ----------

class StateDebouncer:
    """A new state must hold for `delay` seconds before it is shown, so the 60 ms beeps do not flicker 'speaking'.
    'listening' is shown at once (the user just pressed the key)."""

    def __init__(self, delay: float = DEBOUNCE_S, clock: Callable[[], float] = time.monotonic, initial: str = "idle") -> None:
        self.delay, self.clock, self.shown = delay, clock, initial
        self._cand: tuple[str, float] | None = None

    def update(self, raw: str) -> str:
        now = self.clock()
        if raw == self.shown:
            self._cand = None
        elif raw == "listening":
            self.shown, self._cand = raw, None
        else:
            if self._cand is None or self._cand[0] != raw:
                self._cand = (raw, now)
            if now - self._cand[1] >= self.delay:
                self.shown, self._cand = raw, None
        return self.shown


# ---------- the app ----------

class TrayApp:
    def __init__(self, args, create_flow: Callable, tray_factory=tray.Tray, welcome_dir: Path | None = None,
                 smoke: bool = False) -> None:
        self.args = args
        self.smoke = smoke
        self.create_flow = create_flow
        self.home = config.app_dir()
        self.welcome_dir = welcome_dir or self.home
        self.tray = tray_factory(on_activate=threaded(self.open_settings, "activate"),
                                 on_open_settings=threaded(self.open_settings, "open-settings"))
        self.settings_host = None  # None = Settings unavailable (tkinter missing); the tray keeps working
        try:
            from . import settings_window  # tkinter: imported only when the tray app exists, never for the console loop

            self.settings_host = settings_window.Host(
                args.config or config.default_config_path(),
                get_app=lambda: self.app, get_error=lambda: self.error, restart=self.restart_app,
                get_hotkey=lambda: self.hotkey, on_applied=self._on_applied)
        except ImportError as e:
            log.warning("Settings are unavailable: %s", e)
        self.app = self.flow = self.hotkey = None
        self.key_name = "pause"
        self.quit_event = threading.Event()
        self.exit_code = 0
        self.restarting = False
        self.error: str | None = None
        self._status = "Starting..."
        self._download: tuple[str, int] | None = None  # (file name, percent or -1)
        self._ready = False
        self._lock = threading.Lock()
        self._welcome_active = threading.Event()  # the spoken welcome is pending or playing: counts as busy
        self._pack_restart = False  # the automatic restart after the GPU libraries arrived is already scheduled
        self._last_settings = float("-inf")  # monotonic time of the last accepted Settings request

    # ---- life cycle ----
    def run(self) -> int:
        download.set_observer(self._on_download)
        CrashHooks(self.tray.balloon).install()
        self._refresh_menu()
        if self.smoke:
            log.info("Smoke test mode: tray window only, nothing is loaded")
            self._set_state("idle")
            self.tray.set_tooltip("Parley (smoke test)")
            timer = threading.Timer(SMOKE_TIMEOUT_S, self.request_quit)  # never leave a stray process behind
            timer.daemon = True
            timer.start()
        else:
            self._set_state("starting")
            self.tray.set_tooltip("Parley: starting...")
            threading.Thread(target=self._boot, daemon=True, name="boot").start()
        asked = False
        try:
            self.tray.run()
        except KeyboardInterrupt:
            asked = True
        finally:
            # Recorded BEFORE shutdown(), which sets quit_event itself. A window that existed and then closed
            # (Quit, WM_CLOSE, logoff) is a quit; run() returning without ever having created it is a failure.
            asked = asked or self.quit_event.is_set() or bool(getattr(self.tray, "created", True))
            self.shutdown()
        if not asked and not self.exit_code:
            log.error("The tray window could not be created")
            self.exit_code = 1
        if self.restarting:
            self._relaunch()  # an update was installed, or the user chose Restart now in Settings
        return self.exit_code

    def request_quit(self) -> None:
        self.quit_event.set()
        self.tray.stop()

    def shutdown(self) -> None:
        self.quit_event.set()
        download.set_observer(None)
        ui.stop()  # closes the settings window; Tk is destroyed on its own thread
        for stop in (getattr(self.hotkey, "stop", None), getattr(self.flow, "stop", None)):
            if stop:
                try:
                    stop()
                except Exception:
                    log.exception("Shutdown step failed")
        # llama-server is killed by its job object / atexit hook when this process exits

    # ---- boot thread ----
    def _boot(self) -> None:
        try:
            credstore.migrate_legacy_key(self.args.config or config.default_config_path())
            cfg = config.load(self.args.config)
            self.key_name = cfg.hotkey.key
            from .app import Parley
            from .hotkey import HotkeyListener

            app = Parley(cfg, use_mic=True, on_pack_ready=self._on_pack_ready)
            hotkey = HotkeyListener(app, cfg.hotkey.key)
            hotkey.start()
            app.on_applied = self._on_applied
            app.hotkey = hotkey  # Settings re-points it when the talk key changes
            self.app, self.hotkey = app, hotkey
            self.flow = self.create_flow(app, cfg)
        except SystemExit as e:  # a config or setup mistake the user can fix: stay alive and say so
            self._fail(e.code if isinstance(e.code, str) else "Parley could not start.")
            return
        except Exception as e:
            log.exception("Startup failed")
            if _recoverable(e):
                self._fail("Setup failed. Check your internet connection and disk space, then restart Parley.")
            else:
                self._fail("Parley could not start. The log folder has details.", fatal=True)
            return
        log.info("Ready. Hold [%s] and speak; release to send.", self.key_name)
        self._ready = True
        self._download = None
        self._apply_activity("idle")
        self._welcome()
        if self.flow is not None:
            try:
                self.flow.start()
            except Exception:
                log.exception("Update flow could not start")
        self._poll()

    def _fail(self, msg: object, fatal: bool = False) -> None:
        text = short_error(msg)
        log.error("Startup problem: %s", text)
        self.error = text
        self._download = None
        self._status = text[:60]
        self._refresh_menu()
        self._set_state("error")
        self.tray.set_tooltip(("Parley: " + text)[:127])
        self.tray.balloon("Parley could not start", text)
        if fatal:
            self.exit_code = 1
            timer = threading.Timer(4.0, self.tray.stop)  # long enough to read the balloon
            timer.daemon = True  # must not keep the process alive if the window is closed first
            timer.start()

    def _set_state(self, state: str) -> None:
        log.info("Tray icon: %s", state)
        self.tray.set_state(state)

    def _welcome(self) -> threading.Thread | None:
        """First run only (the `tray_welcomed` mark): a balloon, and Parley says a short brief out loud. The mark is
        written once the brief has really been spoken, so an interrupted welcome comes back on the next start."""
        mark = self.welcome_dir / WELCOME_MARK
        if mark.exists():
            return None
        self.tray.balloon("Parley is ready", f"Hold {self._key_label()} and speak.")
        if not hasattr(self.app, "speak_notice"):
            self._mark_welcomed(mark)
            return None
        self._welcome_active.set()
        t = threading.Thread(target=self._speak_welcome, args=(mark,), daemon=True, name="welcome")
        t.start()
        return t

    def _mark_welcomed(self, mark: Path) -> None:
        try:
            mark.parent.mkdir(parents=True, exist_ok=True)
            mark.write_text("1", "utf-8")
        except OSError:
            pass

    def welcome_sentences(self) -> list[str]:
        key = self._key_label()
        return [
            "Welcome to Parley.",
            f"To ask me something, hold the {key} key, speak, and let go when you are done.",
            f"Press {key} again while I am talking to interrupt me.",
            "To change settings, right-click the Parley icon in the taskbar notification area, near the clock, "
            "and choose Settings.",
            "If you don't see the icon, click the little up arrow to find it.",
        ]

    def _speak_welcome(self, mark: Path) -> None:
        try:
            app = self.app
            turn0 = app.turn
            deadline = time.monotonic() + WELCOME_MAX_S
            while app.is_busy():  # wait for a quiet moment; a key press meanwhile cancels the welcome
                if self.quit_event.wait(IDLE_POLL_S) or app.turn != turn0 or time.monotonic() > deadline:
                    return
            if app.turn != turn0:
                return
            turn = app.speak_notice(self.welcome_sentences())
            while not app.finished.wait(IDLE_POLL_S):
                if self.quit_event.is_set() or app.turn != turn or time.monotonic() > deadline:
                    return
            if app.turn == turn:
                self._mark_welcomed(mark)
                log.info("Spoken welcome done")
        except Exception:
            log.exception("The spoken welcome failed")
        finally:
            self._welcome_active.clear()

    # ---- GPU libraries arrived: restart so the models use them ----
    def _on_pack_ready(self) -> None:
        """Called on the download thread after the CUDA pack was installed in THIS run (never when it was already
        there). Restarts Parley once, when idle, so Whisper, Kokoro and the LLM come up on the GPU. Not after an
        update/Settings restart (`--restarted`): that is the loop guard for a pack that installs but never becomes usable."""
        from . import cuda_pack

        if self.args.restarted or cuda_pack.usable_dir(self.home) is None:
            log.info("GPU libraries installed; they are used from the next start")
            return
        with self._lock:
            if self._pack_restart:
                return
            self._pack_restart = True
        log.info("GPU libraries installed; restarting Parley when it is idle")
        quiet = 0
        while not self.quit_event.wait(IDLE_POLL_S):
            app = self.app
            if self._ready and not self.error and app is not None and not self._welcome_active.is_set() and not app.is_busy():
                quiet += 1
            else:
                quiet = 0
            if quiet >= IDLE_CHECKS:
                self.tray.balloon("Parley", "GPU libraries ready - restarting Parley")
                self.restart_app()
                return

    def _key_label(self) -> str:
        return self.key_name.replace("_", " ").title()

    # ---- poll thread ----
    def _poll(self) -> None:
        deb = StateDebouncer()
        shown = "idle"
        while not self.quit_event.wait(POLL_S):
            if self.flow is not None and self.flow.restart_requested.is_set():
                self.restarting = True
                self.request_quit()
                return
            try:
                new = deb.update(self.app.activity())
            except Exception:
                log.debug("activity poll failed", exc_info=True)
                continue
            if new != shown:
                shown = new
                self._apply_activity(new)

    def _apply_activity(self, state: str) -> None:
        if self.error:
            return
        self._status = {"idle": f"Ready - hold {self._key_label()} to talk", "listening": "Listening...",
                        "thinking": "Thinking...", "speaking": "Speaking..."}.get(state, state)
        self._set_state(state)
        self.tray.set_tooltip(f"Parley: hold {self._key_label()} to talk" if state == "idle"
                              else f"Parley: {state}")
        self._refresh_menu()

    # ---- downloads (first run: a few GB) ----
    def _on_download(self, name: str, done: int, total: int) -> None:
        if self._ready or self.error:
            return
        if total and done >= total:
            self._download = None
            self.tray.set_tooltip("Parley: starting...")
            self._status = "Starting..."
        else:
            pct = done * 100 // total if total else -1
            if self._download == (name, pct):
                return
            self._download = (name, pct)
            label = f"Downloading {name[:40]}" + (f" {pct}%" if pct >= 0 else "")
            self._status = label
            self.tray.set_tooltip(f"Parley: {label}")
        self._refresh_menu()

    # ---- menu ----
    def _refresh_menu(self) -> None:
        can_update = self.flow is not None and getattr(self.flow, "updater", None) is not None
        self.tray.set_menu(tray.default_items(
            self._status,
            on_settings=threaded(self.open_settings, "settings"),
            on_check_updates=threaded(self.check_updates, "check-updates"),
            on_open_logs=threaded(self.open_logs, "open-logs"),
            on_quit=threaded(self.request_quit, "quit"),
            can_check_updates=can_update,
        ))

    def open_settings(self) -> None:
        """Menu 'Settings', a left click on the icon, or a second launch's request: open the window or focus the open one."""
        now = time.monotonic()
        with self._lock:
            if now - self._last_settings < SETTINGS_DEBOUNCE_S:
                log.debug("Settings request ignored (repeat within %.1f s)", SETTINGS_DEBOUNCE_S)
                return
            self._last_settings = now
        log.info("Settings requested")
        if self.settings_host is None:
            self.tray.balloon("Parley", "Settings could not start (a component is missing). Edit config.toml for now.")
            return
        from . import settings_window

        if not settings_window.open_settings(self.settings_host):
            self.tray.balloon("Parley", "The settings window could not start. Edit config.toml for now.")

    def _on_applied(self, cfg) -> None:
        """Settings were applied to the running app: keep the tray's key name (tooltip, status line) in step."""
        self.key_name = cfg.hotkey.key
        self._refresh_menu()

    def restart_app(self) -> None:
        """'Restart now' in Settings: quit, then start a fresh copy (it waits for this one's instance lock)."""
        self.restarting = True
        self.request_quit()

    def _relaunch(self) -> None:
        if self.flow is not None:
            self.flow.restart()
            return
        from . import autoupdate

        args = [a for a in sys.argv[1:] if a != autoupdate.RESTARTED_FLAG]
        root = autoupdate.install_root()
        if root is not None:
            autoupdate.spawn_launcher(root, [*args, autoupdate.RESTARTED_FLAG])
            return
        if "--tray" not in args:
            args.append("--tray")
        import subprocess

        flags = getattr(subprocess, "DETACHED_PROCESS", 0) | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        try:
            subprocess.Popen([sys.executable, "-m", "parley", *args, autoupdate.RESTARTED_FLAG], close_fds=True, creationflags=flags)
        except OSError as e:
            log.warning("Cannot restart: %s", e)

    def check_updates(self) -> None:
        if self.flow is not None:
            self.flow.check_now(lambda msg: self.tray.balloon("Parley", msg))

    def open_logs(self) -> None:
        d = logsetup.log_dir(self.home)
        d.mkdir(parents=True, exist_ok=True)
        os.startfile(str(d))  # noqa: S606 - our own folder


def _recoverable(e: Exception) -> bool:
    import httpx

    return isinstance(e, (OSError, httpx.HTTPError, download.DownloadError))


def run(args, create_flow: Callable) -> int:
    """Entry point from `__main__` once logging is set up. Returns the process exit code."""
    if sys.platform != "win32" or not tray.AVAILABLE:
        raise SystemExit("Tray mode needs Windows with pywin32. Run plain `parley` for the console version.")
    home = config.app_dir()
    enable_faulthandler(home)
    handle = instance.acquire(home, instance.RESTART_WAIT_S if args.restarted else 0.0)
    if handle is None:
        woke = instance.wake_existing()
        log.info("Parley is already running (%s)", "asked it to open Settings" if woke else "no window found")
        return 0
    app = TrayApp(args, create_flow, smoke=smoke_requested())
    try:
        return app.run()
    finally:
        del handle  # released by process exit; kept referenced until here
