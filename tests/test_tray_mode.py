"""Tray mode pieces that need no desktop, models or microphone (stdlib unittest).

Run: uv run python -m unittest discover -s tests -v
Scratch dirs are temp dirs named parley_test_traymode_*; the real Parley home is never touched.
"""
import argparse
import logging
import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from parley import __main__ as cli
from parley import logsetup, tray_mode
from parley.app import Parley, activity_of

SECRET = "sk-ant-" "api03-ABCDEFGHIJKLMNOPQRSTUVWXYZ012345"


class Q:
    def __init__(self, n: int = 0) -> None:
        self.unfinished_tasks = n


class ActivityTests(unittest.TestCase):
    TABLE = [  # recording, jobs, sentences, playing -> state
        (False, False, False, False, "idle"),
        (True, False, False, False, "listening"),
        (True, True, True, True, "listening"),  # the start beep is playing while we record
        (False, True, False, False, "thinking"),
        (False, False, True, False, "thinking"),
        (False, True, True, False, "thinking"),
        (False, False, False, True, "speaking"),
        (False, True, True, True, "speaking"),
    ]

    def test_pure_table(self):
        for rec, jobs, sent, play, want in self.TABLE:
            self.assertEqual(activity_of(rec, jobs, sent, play), want, (rec, jobs, sent, play))

    def test_method_reads_queues_and_recorder(self):
        p = object.__new__(Parley)
        p.recorder = SimpleNamespace(recording=False)
        p._jobs, p._sentences, p.player = Q(), Q(), SimpleNamespace(queue=Q())
        self.assertEqual(p.activity(), "idle")
        p._jobs = Q(1)
        self.assertEqual(p.activity(), "thinking")
        p.player.queue = Q(2)
        self.assertEqual(p.activity(), "speaking")
        p.recorder.recording = True
        self.assertEqual(p.activity(), "listening")
        p.recorder = None  # --ask runs have no recorder
        self.assertEqual(p.activity(), "speaking")


class DebounceTests(unittest.TestCase):
    def setUp(self):
        self.t = 0.0
        self.deb = tray_mode.StateDebouncer(0.15, lambda: self.t)

    def feed(self, raw: str, dt: float) -> str:
        self.t += dt
        return self.deb.update(raw)

    def test_short_beep_does_not_show_speaking(self):
        self.assertEqual(self.feed("speaking", 0.0), "idle")  # a 60 ms beep seen by one 100 ms poll
        self.assertEqual(self.feed("thinking", 0.1), "idle")
        self.assertEqual(self.feed("thinking", 0.1), "idle")
        self.assertEqual(self.feed("thinking", 0.1), "thinking")  # held 0.2 s

    def test_sustained_state_is_shown_after_delay(self):
        self.feed("speaking", 0.0)
        self.assertEqual(self.feed("speaking", 0.1), "idle")
        self.assertEqual(self.feed("speaking", 0.1), "speaking")

    def test_listening_is_immediate(self):
        self.assertEqual(self.feed("listening", 0.0), "listening")
        self.assertEqual(self.feed("idle", 0.1), "listening")
        self.assertEqual(self.feed("idle", 0.1), "listening")
        self.assertEqual(self.feed("idle", 0.1), "idle")


class ThreadedTests(unittest.TestCase):
    def test_slow_callback_does_not_block_the_caller(self):
        gate, started, done = threading.Event(), threading.Event(), threading.Event()

        def slow():
            started.set()
            gate.wait(10)
            done.set()

        t0 = time.monotonic()
        tray_mode.threaded(slow)()
        self.assertLess(time.monotonic() - t0, 0.5)
        self.assertTrue(started.wait(5))
        self.assertFalse(done.is_set())
        gate.set()
        self.assertTrue(done.wait(5))

    def test_failure_is_logged_not_raised(self):
        with self.assertLogs("parley.tray_mode", "ERROR"):
            tray_mode.threaded(lambda: 1 / 0, "boom")()
            time.sleep(0.5)


class FakeTray:
    def __init__(self, **kw) -> None:
        self.kw, self.calls, self.items = kw, [], []

    def set_state(self, s): self.calls.append(("state", s))
    def set_tooltip(self, t): self.calls.append(("tip", t))
    def balloon(self, title, text): self.calls.append(("balloon", title, text))
    def set_menu(self, items): self.items = list(items)
    def stop(self): self.calls.append(("stop",))
    def run(self): pass


class GoneTray(FakeTray):
    """run() returns at once, as the real Tray does when its window could not be created."""
    created = False


class ClosedTray(FakeTray):
    """The window existed and was closed from outside (WM_CLOSE / logoff): quit_event was never set by the app."""
    created = True


class ScratchHome(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="parley_test_traymode_")
        self.home = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        env = mock.patch.dict(os.environ, {"PARLEY_HOME": str(self.home)})
        env.start()
        self.addCleanup(env.stop)


class LoggingTests(ScratchHome):
    def setUp(self):
        super().setUp()
        root = logging.getLogger()
        handlers, level = root.handlers[:], root.level
        self.addCleanup(lambda: (setattr(root, "handlers", handlers), root.setLevel(level)))

    def test_file_log_is_rotating_and_redacted(self):
        listener = logsetup.setup_file_logging(self.home)
        fh = listener.handlers[0]
        self.assertEqual((fh.maxBytes, fh.backupCount), (1 << 20, 3))
        logging.getLogger("parley.x").error("auth failed for %s", SECRET)
        try:
            raise RuntimeError("boom " + SECRET)
        except RuntimeError:
            logging.getLogger("parley.x").exception("failed")
        listener.stop()
        fh.close()
        text = (self.home / "logs" / "parley.log").read_text("utf-8")
        self.assertIn("auth failed for", text)
        self.assertIn("RuntimeError", text)
        self.assertNotIn(SECRET, text)

    def test_crash_hooks_log_redacted_and_balloon_once(self):
        listener = logsetup.setup_file_logging(self.home)
        shown = []
        hooks = tray_mode.CrashHooks(lambda title, text: shown.append((title, text)))
        try:
            raise ValueError("bad " + SECRET)
        except ValueError:
            info = sys.exc_info()
        hooks.on_exception(*info)
        hooks.on_thread_exception(SimpleNamespace(exc_type=info[0], exc_value=info[1], exc_traceback=info[2],
                                                  thread=SimpleNamespace(name="worker")))
        hooks.on_thread_exception(SimpleNamespace(exc_type=SystemExit, exc_value=None, exc_traceback=None, thread=None))
        listener.stop()
        listener.handlers[0].close()
        self.assertEqual(len(shown), 1)  # one balloon per session
        self.assertNotIn("sk-", shown[0][1])
        text = (self.home / "logs" / "parley.log").read_text("utf-8")
        self.assertIn("Unhandled exception in thread worker", text)
        self.assertNotIn(SECRET, text)


class TrayAppTests(ScratchHome):
    def make(self):
        args = argparse.Namespace(config=None, restarted=False, verbose=False)
        return tray_mode.TrayApp(args, lambda app, cfg: None, tray_factory=FakeTray, welcome_dir=self.home)

    def test_startup_error_is_short_redacted_and_keeps_running(self):
        app = self.make()
        app._fail(f"Bad key {SECRET} " + "x" * 400)
        calls = app.tray.calls
        self.assertIn(("state", "error"), calls)
        balloon = next(c for c in calls if c[0] == "balloon")
        self.assertLessEqual(len(balloon[2]), 200)
        self.assertNotIn("sk-ant", balloon[2])
        self.assertNotIn(("stop",), calls)  # stays alive so the setting can be fixed
        self.assertEqual(app.exit_code, 0)

    def test_menu_callbacks_are_handed_to_workers(self):
        app = self.make()
        gate = threading.Event()
        app.open_settings = lambda: gate.wait(10)
        app._refresh_menu()
        settings = next(i for i in app.tray.items if i.id == "settings")
        t0 = time.monotonic()
        settings.callback()
        self.assertLess(time.monotonic() - t0, 0.5)
        gate.set()

    def test_status_line_is_disabled_and_updates_check_hidden_without_flow(self):
        app = self.make()
        app._refresh_menu()
        ids = [i.id for i in app.tray.items]
        self.assertFalse(app.tray.items[0].enabled)
        self.assertNotIn("check_updates", ids)

    def test_download_shows_percent_in_tooltip_and_menu(self):
        app = self.make()
        app._on_download("model.gguf", 430, 1000)
        self.assertIn(("tip", "Parley: Downloading model.gguf 43%"), app.tray.calls)
        self.assertEqual(app.tray.items[0].label, "Downloading model.gguf 43%")
        app._on_download("model.gguf", 1000, 1000)
        self.assertEqual(app.tray.items[0].label, "Starting...")

    def run_app(self, factory, smoke=True):
        args = argparse.Namespace(config=None, restarted=False, verbose=False)
        app = tray_mode.TrayApp(args, lambda a, c: None, tray_factory=factory, welcome_dir=self.home, smoke=smoke)
        with mock.patch.object(sys, "excepthook", sys.excepthook),                 mock.patch.object(threading, "excepthook", threading.excepthook),                 mock.patch.object(tray_mode.threading, "Timer"):
            if factory is GoneTray:
                with self.assertLogs("parley.tray_mode", "ERROR"):
                    return app.run()
            return app.run()

    def test_run_exits_nonzero_when_the_tray_window_could_not_be_created(self):
        self.assertEqual(self.run_app(GoneTray), 1)  # the launcher's rollback relies on a non-zero exit

    def test_run_exits_zero_when_the_window_was_closed_from_outside(self):
        self.assertEqual(self.run_app(ClosedTray), 0)

    def test_run_exits_zero_after_a_quit_request(self):
        class QuitTray(GoneTray):
            def run(self):
                self.owner.request_quit()
        args = argparse.Namespace(config=None, restarted=False, verbose=False)
        holder = {}
        def factory(**kw):
            t = QuitTray(**kw)
            holder["t"] = t
            return t
        app = tray_mode.TrayApp(args, lambda a, c: None, tray_factory=factory, welcome_dir=self.home, smoke=True)
        holder["t"].owner = app
        with mock.patch.object(sys, "excepthook", sys.excepthook), mock.patch.object(threading, "excepthook", threading.excepthook),                 mock.patch.object(tray_mode.threading, "Timer"):
            self.assertEqual(app.run(), 0)

    def test_fatal_error_timer_is_a_daemon(self):
        app = self.make()
        with mock.patch.object(tray_mode.threading, "Timer") as timer:
            app._fail("boom", fatal=True)
        self.assertEqual(app.exit_code, 1)
        self.assertIs(timer.return_value.daemon, True)
        timer.return_value.start.assert_called_once()
        self.assertEqual(timer.call_args.args[1], app.tray.stop)

    def test_smoke_mode_loads_nothing(self):
        app = self.make()
        app.smoke = True
        with mock.patch.object(tray_mode.threading, "Thread") as thread, mock.patch.object(tray_mode.threading, "Timer"),                 mock.patch.object(sys, "excepthook", sys.excepthook), mock.patch.object(threading, "excepthook", threading.excepthook):
            app.run()
        thread.assert_not_called()  # no boot thread: no Parley, no models, no downloads
        self.assertIn(("state", "idle"), app.tray.calls)

    def test_quit_stops_the_tray(self):
        app = self.make()
        app.request_quit()
        self.assertIn(("stop",), app.tray.calls)
        self.assertTrue(app.quit_event.is_set())

    def test_welcome_balloon_only_once(self):
        app = self.make()
        app._welcome()
        app._welcome()
        balloons = [c for c in app.tray.calls if c[0] == "balloon"]
        self.assertEqual(len(balloons), 1)
        self.assertIn("Pause", balloons[0][2])


class FakeApp:
    """Stands in for Parley: a busy flag, a turn counter and the `speak_notice` / `finished` pair."""
    def __init__(self) -> None:
        self.turn, self.busy, self.spoken = 1, False, []
        self.finished = threading.Event()

    def is_busy(self): return self.busy

    def speak_notice(self, sentences):
        self.spoken.append(list(sentences))
        self.finished.clear()
        return self.turn


class WelcomeTests(ScratchHome):
    make = TrayAppTests.make
    def setUp(self):
        super().setUp()
        p = mock.patch.object(tray_mode, "IDLE_POLL_S", 0.01)
        p.start()
        self.addCleanup(p.stop)
        self.app = self.make()
        self.app.app = FakeApp()
        self.mark = self.home / tray_mode.WELCOME_MARK

    def test_spoken_with_key_name_settings_hint_and_marked_only_afterwards(self):
        t = self.app._welcome()
        time.sleep(0.2)
        self.assertTrue(self.app._welcome_active.is_set())  # counts as busy while it plays
        self.assertFalse(self.mark.exists())
        text = " ".join(self.app.app.spoken[0])
        self.assertIn("Pause", text)
        self.assertIn("right-click the Parley icon", text)
        self.assertIn("Settings", text)
        self.assertIn("up arrow", text)
        self.assertIn("interrupt", text)
        self.assertIn(len(text.split()), range(40, 75))  # about 20 seconds
        self.app.app.finished.set()
        t.join(5)
        self.assertTrue(self.mark.exists())
        self.assertFalse(self.app._welcome_active.is_set())
        balloons = [c for c in self.app.tray.calls if c[0] == "balloon"]
        self.assertEqual(len(balloons), 1)  # the old balloon stays
        self.assertIsNone(self.app._welcome())  # never again
        self.assertEqual(len(self.app.app.spoken), 1)

    def test_key_press_while_speaking_leaves_it_unmarked(self):
        t = self.app._welcome()
        time.sleep(0.2)
        self.app.app.turn += 1  # the user pressed the talk key
        t.join(5)
        self.assertFalse(self.mark.exists())
        self.assertFalse(self.app._welcome_active.is_set())

    def test_key_press_before_it_starts_cancels_it(self):
        self.app.app.busy = True
        t = self.app._welcome()
        time.sleep(0.1)
        self.app.app.turn += 1
        t.join(5)
        self.assertEqual(self.app.app.spoken, [])
        self.assertFalse(self.mark.exists())

    def test_waits_until_idle_before_speaking(self):
        self.app.app.busy = True
        t = self.app._welcome()
        time.sleep(0.1)
        self.assertEqual(self.app.app.spoken, [])
        self.app.app.busy = False
        time.sleep(0.2)
        self.assertEqual(len(self.app.app.spoken), 1)
        self.app.app.finished.set()
        t.join(5)

    def test_without_a_speaker_it_is_the_balloon_alone(self):
        self.app.app = None
        self.assertIsNone(self.app._welcome())
        self.assertTrue(self.mark.exists())


class PackRestartTests(ScratchHome):
    make = TrayAppTests.make
    def setUp(self):
        super().setUp()
        p = mock.patch.object(tray_mode, "IDLE_POLL_S", 0.01)
        p.start()
        self.addCleanup(p.stop)
        self.app = self.make()
        self.app.app, self.app._ready = FakeApp(), True
        self.restarts = []
        self.app.restart_app = lambda: self.restarts.append(1)
        u = mock.patch("parley.cuda_pack.usable_dir", return_value=self.home)
        self.usable = u.start()
        self.addCleanup(u.stop)

    def test_restarts_once_when_idle_with_a_balloon(self):
        self.app._on_pack_ready()
        self.app._on_pack_ready()
        self.assertEqual(self.restarts, [1])
        self.assertIn(("balloon", "Parley", "GPU libraries ready - restarting Parley"), self.app.tray.calls)

    def test_waits_while_busy_or_welcoming(self):
        self.app.app.busy = True
        t = threading.Thread(target=self.app._on_pack_ready, daemon=True)
        t.start()
        time.sleep(0.15)
        self.assertEqual(self.restarts, [])
        self.app.app.busy = False
        self.app._welcome_active.set()
        time.sleep(0.15)
        self.assertEqual(self.restarts, [])
        self.app._welcome_active.clear()
        t.join(5)
        self.assertEqual(self.restarts, [1])

    def test_no_restart_after_a_restart_or_without_a_usable_pack_or_when_quitting(self):
        self.app.args.restarted = True  # loop guard
        self.app._on_pack_ready()
        self.app.args.restarted = False
        self.usable.return_value = None
        self.app._on_pack_ready()
        self.usable.return_value = self.home
        self.app.quit_event.set()
        self.app._on_pack_ready()
        self.assertEqual(self.restarts, [])


class PackRestartExtraTests(PackRestartTests):
    def test_busy_again_right_before_restart_goes_back_to_polling(self):
        calls = {"n": 0}
        app = self.app.app

        def busy():  # idle for the 3 polls, busy at the final look, then idle again
            calls["n"] += 1
            return calls["n"] == 4
        app.is_busy = busy
        self.app._on_pack_ready()
        self.assertEqual(self.restarts, [1])
        self.assertGreater(calls["n"], 7)  # the busy look reset the count: 3 more idle polls + final look

    def _boot_with(self, restarted):
        self.app.args.restarted = restarted
        seen = {}

        class P:
            def __init__(me, cfg, use_mic=True, on_pack_ready=None):
                seen["hook"] = on_pack_ready
                raise SystemExit("stop here")
        with mock.patch("parley.app.Parley", P), mock.patch("parley.config.load", return_value=SimpleNamespace(
                hotkey=SimpleNamespace(key="pause"))), mock.patch("parley.credstore.migrate_legacy_key"):
            self.app._boot()
        return seen["hook"]

    def test_boot_wires_the_hook_only_when_not_restarted(self):
        self.assertEqual(self._boot_with(False), self.app._on_pack_ready)
        self.assertIsNone(self._boot_with(True))

    def test_wording_differs_without_the_hook(self):
        from parley import cuda_pack
        said = {}
        for name, hook in (("with", lambda: None), ("without", None)):
            lines = []
            with tempfile.TemporaryDirectory(prefix="parley_test_wording_") as d,                     mock.patch.object(cuda_pack, "install"), mock.patch.object(cuda_pack, "load_manifest"),                     mock.patch.object(cuda_pack, "usable_dir", return_value=None),                     mock.patch.object(cuda_pack, "needed", return_value=True):
                cuda_pack.maybe_download(None, None, Path(d), lines.append, frozen=True, on_installed=hook)
            said[name] = lines[-1]
        self.assertIn("restart in a moment", said["with"])
        self.assertIn("next time it starts", said["without"])


class SpeakNoticeTests(unittest.TestCase):
    def test_queues_the_sentences_then_the_end_marker_in_one_turn(self):
        import queue
        p = object.__new__(Parley)
        p.turn, p._sentences, p.finished = 7, queue.Queue(), threading.Event()
        p.finished.set()
        self.assertEqual(p.speak_notice(["A.", "B."]), 7)
        self.assertFalse(p.finished.is_set())
        items = [p._sentences.get_nowait() for _ in range(3)]
        self.assertEqual([(t, s) for t, s, _ in items], [(7, "A."), (7, "B."), (7, None)])


class SmokeGateTests(unittest.TestCase):
    def test_needs_the_variable_and_a_throwaway_home(self):
        ok = {"PARLEY_TRAY_SMOKE": "1", "PARLEY_HOME": "C:/tmp/parley_test_x"}
        self.assertTrue(tray_mode.smoke_requested(ok))
        for bad in ({**ok, "PARLEY_TRAY_SMOKE": "0"}, {**ok, "PARLEY_TRAY_SMOKE": "true"},
                    {"PARLEY_TRAY_SMOKE": "1"},  # no PARLEY_HOME: the real home
                    {**ok, "PARLEY_HOME": "C:/Users/me/AppData/Local/parley"},
                    {**ok, "PARLEY_HOME": ""}, {"PARLEY_HOME": ok["PARLEY_HOME"]}):
            self.assertFalse(tray_mode.smoke_requested(bad), bad)


class ShimTests(unittest.TestCase):
    def test_app_imports_and_logging_work_with_no_stdout_stderr(self):
        with tempfile.TemporaryDirectory(prefix="parley_test_shim_") as d:
            code = "\n".join([
                "import sys",
                "sys.stdout = None; sys.stderr = None",
                "from parley import logsetup",
                "assert logsetup.install_stdio_shim()",
                "from pathlib import Path",
                f"l = logsetup.setup_file_logging(Path({d!r}))",
                "print('hello from print'); sys.stderr.write('partial'); sys.stderr.write(' line\\n')",
                "assert sys.stderr.isatty() is False",
                "from parley import download, __main__",
                "import argparse",
                "try:",
                "    argparse.ArgumentParser(prog='x').parse_args(['--nope'])",
                "except SystemExit as e:",
                "    assert e.code == 2",
                "l.stop()",
            ])
            r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120,
                               env={**os.environ, "PARLEY_HOME": d})
            self.assertEqual(r.returncode, 0, r.stderr[-500:])
            log = (Path(d) / "logs" / "parley.log").read_text("utf-8")
            self.assertIn("hello from print", log)
            self.assertIn("partial line", log)
            self.assertIn("unrecognized arguments", log)

    def test_shim_leaves_real_streams_alone(self):
        self.assertFalse(logsetup.install_stdio_shim())


class ConsoleLoopTests(unittest.TestCase):
    def test_console_loop_uses_listener_and_honours_stop(self):
        events = []

        class FakeListener:
            def __init__(self, app, key): events.append(("init", key))
            def __enter__(self): events.append("enter"); return self
            def __exit__(self, *a): events.append("exit")
            def wait(self, stop): events.append(("wait", stop))

        stop = threading.Event()
        with mock.patch.object(cli.hotkey, "HotkeyListener", FakeListener):
            cli.run_hotkey_loop(SimpleNamespace(), "pause", lambda: events.append("ready"), stop)
        self.assertEqual(events, [("init", "pause"), "enter", "ready", ("wait", stop), "exit"])

    def test_plain_python_is_not_the_tray_exe(self):
        self.assertFalse(tray_mode.implied())


if __name__ == "__main__":
    unittest.main()
