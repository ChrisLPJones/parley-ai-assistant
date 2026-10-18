"""The Settings window: pure helpers, and the real window against the ui_preview stub app (skipped where Tk can't start).

Run: uv run python -m unittest discover -s tests -v
Scratch dirs are temp dirs named parley_test_settingswin_*; the real Parley home is never touched.
"""
import argparse
import logging
import os
import stat
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from parley import config_io, settings_schema as S, settings_window as sw, ui, ui_preview, voices

SECRET = "sk-ant-" "api03-ABCDEFGHIJKLMNOPQRSTUVWXYZ012345"
ROOT = Path(__file__).resolve().parent.parent


def tray_mode_debounce() -> float:
    from parley import tray_mode

    return tray_mode.SETTINGS_DEBOUNCE_S


class PureTests(unittest.TestCase):
    def test_summarize(self):
        self.assertEqual(sw.summarize(["tts.voice"], []), ("ok", "Applied."))
        self.assertEqual(sw.summarize([], []), ("ok", "Applied."))
        kind, text = sw.summarize(["tts.voice"], ["stt.model"])
        self.assertEqual((kind, text), ("warn", "Saved. 1 setting needs a restart (Speech model)."))
        kind, text = sw.summarize([], ["stt.model", "tts.device", "gpu.download_cuda"])
        self.assertIn("3 settings need a restart", text)
        self.assertIn("Voice device", text)
        self.assertEqual(sw.summarize([], [], ["hotkey.key"]), ("ok", "Saved. 1 setting will apply after this answer."))
        self.assertIn("next time Parley starts", sw.summarize(["updates.check"], [])[1])

    def test_restart_banner_names_at_most_six_then_counts_the_rest(self):
        many = [s.path for s in S.SETTINGS if s.apply == S.RESTART][:6] + ["llm.backend", "llm.preset", "tools.enabled"]
        text = sw.summarize([], many)[1]
        self.assertIn("9 settings need a restart", text)
        self.assertIn("and 3 more", text)
        self.assertNotIn("more", sw.summarize([], many[:6])[1])

    def test_device_choices(self):
        labels, mapping, cur = sw.device_choices(None, ["Mic A", "Mic B"])
        self.assertEqual((labels, cur), ([sw.SYSTEM_DEFAULT, "Mic A", "Mic B"], sw.SYSTEM_DEFAULT))
        self.assertIsNone(mapping[sw.SYSTEM_DEFAULT])
        self.assertEqual(sw.device_choices("Mic B", ["Mic A", "Mic B"])[2], "Mic B")
        labels, mapping, cur = sw.device_choices("Old USB", ["Mic A"])  # a configured device that is not plugged in
        self.assertEqual(cur, "Old USB (not found)")
        self.assertEqual(mapping[cur], "Old USB")  # not silently replaced
        labels, mapping, cur = sw.device_choices(3, [])
        self.assertEqual((cur, mapping[cur]), ("Device 3", 3))

    def test_voice_groups_and_labels(self):
        groups = voices.group_voices(["bm_george", "af_heart", "zf_xiaobei", "af_bella", "xx_odd", "am_adam"])
        self.assertEqual([g for g, _ in groups], ["American English, female", "American English, male", "British English, male",
                                                  "Mandarin Chinese, female", "Other"])
        self.assertEqual(groups[0][1], [("af_bella", "Bella"), ("af_heart", "Heart")])
        self.assertEqual(voices.friendly_name("af_heart"), "Heart")
        self.assertIn("Heart", voices.sample_text("af_heart"))
        every = voices.group_voices(ui_preview.KOKORO_VOICES)
        self.assertEqual(sum(len(v) for _, v in every), len(ui_preview.KOKORO_VOICES))
        self.assertNotIn("Other", [g for g, _ in every])

    def test_every_control_has_a_schema_entry_with_label_and_help(self):
        for path in (*sw.GENERAL_FIELDS, *sw.VOICE_FIELDS):
            if path.startswith("@"):
                continue
            s = S.BY_PATH[path]
            self.assertTrue(s.label and s.help, path)

    def test_device_pickers_have_no_redundant_help(self):
        # The window (headless of Tk) builds help only when show_help; the two pickers pass show_help=False
        import inspect

        src = inspect.getsource(sw.SettingsWindow._device_row)
        self.assertIn("show_help=False", src)

    def test_no_page_offers_the_api_key(self):
        self.assertNotIn("llm.api_key", (*sw.GENERAL_FIELDS, *sw.VOICE_FIELDS))


def tk_ok() -> bool:
    try:
        return ui.start()
    except Exception:
        return False


class WindowCase(unittest.TestCase):
    """A real window on the UI thread. Tk is only touched through on_ui()."""

    @classmethod
    def setUpClass(cls):
        if not tk_ok():
            raise unittest.SkipTest(f"Tk cannot start here ({ui.last_error()})")
        cls._saved_home = os.environ.get("PARLEY_HOME")

    @classmethod
    def tearDownClass(cls):
        ui.stop()
        if cls._saved_home is None:
            os.environ.pop("PARLEY_HOME", None)
        else:
            os.environ["PARLEY_HOME"] = cls._saved_home

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="parley_test_settingswin_")
        self.addCleanup(self._tmp.cleanup)
        self.home = Path(self._tmp.name)
        self.host, _home, _ = ui_preview.build_host(home=self.home)
        self.cfg_path = self.host.config_path
        self.app = self.host.app
        self.win = None
        self.addCleanup(self.close)

    def close(self):
        if self.win is not None:
            try:
                ui.call(lambda: self.win.close() if self.win.alive() else None)
            except Exception:
                pass

    def open(self, host=None):
        self.win = ui.call(lambda: sw.SettingsWindow(ui.root(), host or self.host))
        return self.win

    def on_ui(self, fn):
        return ui.call(fn)

    def banner(self):
        return self.on_ui(lambda: (self.win.banner.cget("text"), self.win._banner_kind))

    def wait_banner(self, startswith, timeout=5):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            text, kind = self.banner()
            if text.startswith(startswith):
                return text, kind
            time.sleep(0.05)
        self.fail(f"banner never started with {startswith!r}; last: {self.banner()}")

    def save_disabled(self):
        return self.on_ui(lambda: self.win.save_btn.instate(["disabled"]))

    def set_speed(self, v):
        def go():
            self.win.speed_var.set(v)
            self.win._changed("tts.speed")
        self.on_ui(go)


class WindowTests(WindowCase):
    def test_opens_with_the_values_from_config_toml(self):
        self.open()
        st = self.on_ui(lambda: (self.win.key_name, self.win.voice, self.win.speed_var.get(), self.win.page,
                                 self.win.autostart_var.get()))
        self.assertEqual(st, ("pause", "af_heart", 1.1, "general", False))

    def test_open_is_idempotent_and_refocuses(self):
        host = self.host
        sw.open_settings(host)
        end = time.monotonic() + 5
        while self.on_ui(lambda: sw._current is None) and time.monotonic() < end:
            time.sleep(0.05)
        first = self.on_ui(lambda: sw._current)
        self.win = first
        for _ in range(3):
            sw.open_settings(host, "voice")
        time.sleep(0.4)
        self.assertIs(self.on_ui(lambda: sw._current), first)  # still the one window
        self.assertEqual(self.on_ui(lambda: first.page), "voice")
        tops = self.on_ui(lambda: [w for w in ui.root().winfo_children() if w.winfo_class() == "Toplevel"])
        self.assertEqual(len(tops), 1)

    def test_invalid_input_shows_a_message_and_blocks_save(self):
        self.open()
        self.set_speed(5.0)
        err = self.on_ui(lambda: self.win.errors.get("tts.speed"))
        self.assertIn("between 0.5 and 2", err)  # the schema's own sentence
        self.assertTrue(self.save_disabled())
        self.on_ui(self.win.save)
        self.assertEqual(self.banner()[1], "error")
        self.assertNotIn("5.0", self.cfg_path.read_text("utf-8"))
        self.set_speed(1.5)  # fixed: the message goes, Save is back
        self.assertNotIn("tts.speed", self.on_ui(lambda: dict(self.win.errors)))
        self.assertFalse(self.save_disabled())

    def test_an_unknown_key_is_refused_with_the_schema_message(self):
        self.open()

        def go():
            self.win.key_name = "mro"  # an attribute of pynput's Key class, not a key
            self.win._changed("hotkey.key")
            return self.win.errors.get("hotkey.key")

        self.assertIn("Talk key", self.on_ui(go))
        self.assertTrue(self.save_disabled())

    def test_save_writes_only_changed_keys_and_keeps_comments(self):
        self.open()
        self.set_speed(1.5)
        self.on_ui(self.win.save)
        text, kind = self.wait_banner("Applied")
        self.assertEqual(kind, "ok")
        out = self.cfg_path.read_text("utf-8")
        self.assertIn("speed = 1.5", out)
        self.assertIn("# my favourite", out)  # comments survive
        self.assertIn('voice = "af_heart"', out)
        self.assertNotIn("audio", out)  # untouched sections are not written
        self.assertEqual(self.app.cfg.tts.speed, 1.1)  # the stub never mutates; apply_config was asked about the new value

    def test_changing_the_voice_also_sets_its_language(self):
        self.open()
        self.on_ui(lambda: (self.win.voice_tree.selection_set("bf_emma")))
        time.sleep(0.2)
        changes, ok = self.on_ui(self.win.collect)
        self.assertTrue(ok)
        self.assertEqual(changes, {"tts.voice": "bf_emma", "tts.lang": "en-gb"})

    def test_restart_needed_banner_and_button(self):
        host, _h, _ = ui_preview.build_host("restart-needed", home=self.home)
        self.open(host)
        text, kind = self.banner()
        self.assertEqual((text, kind), ("2 settings need a restart (Speech model, Voice device).", "warn"))
        self.assertTrue(self.on_ui(lambda: self.win.restart_btn.winfo_manager() != ""))
        called = []
        host._restart = lambda: called.append(True)
        self.on_ui(self.win._restart_now)
        end = time.monotonic() + 3
        while not called and time.monotonic() < end:
            time.sleep(0.05)
        self.assertEqual(called, [True])

    def test_saved_restart_setting_shows_the_banner_after_save(self):
        self.open()
        # a setting the stub classifies as restart-needed, through the same path the window uses after Save
        self.app.apply_config = lambda new: (["tts.voice"], ["stt.model"])
        self.app._pending = ["stt.model"]
        self.set_speed(1.3)
        self.on_ui(self.win.save)
        text, kind = self.wait_banner("Saved. 1 setting needs a restart")
        self.assertEqual(kind, "warn")
        self.assertTrue(self.on_ui(lambda: self.win.restart_btn.winfo_manager() != ""))

    def test_startup_error_is_shown_on_open(self):
        host, _h, _ = ui_preview.build_host("error", home=self.home)
        self.open(host)
        text, kind = self.banner()
        self.assertEqual(kind, "error")
        self.assertTrue(text.startswith("Parley could not start:"))

    def test_without_a_running_app_save_still_writes_and_the_sample_is_off(self):
        host = sw.Host(self.cfg_path, autostart_api=ui_preview.FakeAutostart(), list_devices=lambda k: [])
        self.open(host)
        self.on_ui(lambda: self.win._poll)  # (poll is scheduled by the window itself)
        self.set_speed(1.25)
        self.on_ui(self.win.save)
        self.assertIn("apply when Parley has started", self.banner()[0])
        self.assertIn("speed = 1.25", self.cfg_path.read_text("utf-8"))
        time.sleep(0.4)
        self.assertTrue(self.on_ui(lambda: self.win.sample_btn.instate(["disabled"])))

    # ---- sample button ----
    def test_sample_button_follows_busy_state_and_plays_the_selection(self):
        self.open()
        self.on_ui(lambda: self.win.show_page("voice"))
        self.on_ui(lambda: self.win.voice_tree.selection_set("am_adam"))
        time.sleep(0.4)
        self.assertFalse(self.on_ui(lambda: self.win.sample_btn.instate(["disabled"])))
        self.on_ui(self.win.sample_btn.invoke)
        end = time.monotonic() + 3
        while not self.app.played and time.monotonic() < end:
            time.sleep(0.05)
        self.assertEqual(self.app.played, [("am_adam", 1.1)])  # the unsaved selection, not the saved voice
        time.sleep(0.5)
        self.assertTrue(self.on_ui(lambda: self.win.sample_btn.instate(["disabled"])))  # busy while it "plays"

    # ---- hotkey capture ----
    def test_press_a_key_sets_the_talk_key(self):
        self.open()
        self.on_ui(self.win._start_capture)  # the preview host's fake listener "presses" F13 after a second
        end = time.monotonic() + 5
        while self.on_ui(lambda: self.win.key_name) != "f13" and time.monotonic() < end:
            time.sleep(0.05)
        self.assertEqual(self.on_ui(lambda: self.win.key_name), "f13")
        self.assertEqual(self.on_ui(lambda: self.win.capture_btn.cget("text")), "Press a key")
        self.assertIsNone(self.host._get_hotkey().capture)  # capture mode is off again: the key works normally

    def test_capture_escape_cancels_and_unusable_keys_are_explained(self):
        self.open()
        hk = self.host._get_hotkey()
        hk.capture = None
        captured = []
        self.host.capture_key = lambda cb: captured.append(cb) or True  # drive the callback by hand
        self.on_ui(self.win._start_capture)
        captured[0](0x1B)  # Escape
        time.sleep(0.3)
        self.assertEqual(self.on_ui(lambda: self.win.key_name), "pause")
        self.on_ui(self.win._start_capture)
        captured[1](0x6B)  # numpad plus: no config name
        time.sleep(0.3)
        self.assertIn("can't be used", self.on_ui(lambda: self.win.errors.get("hotkey.key")))
        self.assertEqual(self.on_ui(lambda: self.win.key_name), "pause")

    def test_closing_cancels_capture(self):
        self.open()
        hk = self.host._get_hotkey()
        self.on_ui(self.win._start_capture)
        self.assertIsNotNone(hk.capture)
        self.on_ui(self.win.close)
        self.assertIsNone(hk.capture)

    # ---- Start with Windows ----
    def test_autostart_is_saved_through_the_host(self):
        self.open()
        self.on_ui(lambda: self.win.autostart_var.set(True))
        self.on_ui(self.win.save)
        self.assertTrue(self.host.autostart.status().enabled)
        self.assertEqual(self.banner()[0], "Applied.")  # no config change, so no apply, no restart

    def test_autostart_failure_is_reported_next_to_the_control(self):
        class Refusing(ui_preview.FakeAutostart):
            def set_enabled(self, on):
                raise OSError("Windows refused the change.")

        host, _h, _ = ui_preview.build_host(home=self.home)
        host.autostart = Refusing()
        self.open(host)
        self.on_ui(lambda: self.win.autostart_var.set(True))
        self.on_ui(self.win.save)
        self.assertIn("Start with Windows failed", self.banner()[0])
        self.assertIn("refused", self.on_ui(lambda: self.win.autostart_error.cget("text")))

    def test_autostart_control_is_disabled_in_a_source_checkout(self):
        host, _h, _ = ui_preview.build_host(home=self.home)
        host.autostart = ui_preview.FakeAutostart(available=False)
        self.open(host)
        self.assertTrue(self.on_ui(lambda: self.win.autostart_cb.instate(["disabled"])))


class SaveFailureTests(WindowCase):
    def test_read_only_config_gives_a_friendly_message(self):
        self.open()
        self.set_speed(1.4)
        os.chmod(self.cfg_path, stat.S_IREAD)
        self.addCleanup(os.chmod, self.cfg_path, stat.S_IWRITE | stat.S_IREAD)
        before = self.cfg_path.read_text("utf-8")
        self.on_ui(self.win.save)
        text, kind = self.banner()
        self.assertEqual(kind, "error")
        self.assertIn("couldn't write config.toml", text)
        self.assertNotIn("Traceback", text)
        self.assertEqual(self.cfg_path.read_text("utf-8"), before)

    def test_oserror_from_the_writer_is_caught(self):
        self.open()
        self.set_speed(1.4)
        with mock.patch.object(config_io, "save_changes", side_effect=PermissionError(13, "Access is denied")):
            self.on_ui(self.win.save)
        text, kind = self.banner()
        self.assertEqual(kind, "error")
        self.assertIn("Access is denied", text)

    def test_config_file_error_text_is_shown(self):
        self.open()
        self.set_speed(1.4)
        self.cfg_path.write_text("this is [not toml", "utf-8")  # broken after the window opened
        self.on_ui(self.win.save)
        text, kind = self.banner()
        self.assertEqual(kind, "error")
        self.assertIn("won't overwrite", text)
        self.assertEqual(self.cfg_path.read_text("utf-8"), "this is [not toml")

    def test_a_corrupt_file_at_open_blocks_saving_and_says_why(self):
        self.cfg_path.write_text("[tts\nvoice = 1", "utf-8")
        self.open()
        text, kind = self.banner()
        self.assertEqual(kind, "error")
        self.assertIn("has a mistake", text)
        self.assertTrue(self.save_disabled())
        self.on_ui(self.win.save)
        self.assertEqual(self.cfg_path.read_text("utf-8"), "[tts\nvoice = 1")

    def test_failures_never_touch_the_running_app(self):
        self.open()
        self.set_speed(1.4)
        calls = []
        self.app.apply_config = lambda new: calls.append(new) or ([], [])
        with mock.patch.object(config_io, "save_changes", side_effect=PermissionError(13, "Access is denied")):
            self.on_ui(self.win.save)
        time.sleep(0.3)
        self.assertEqual(calls, [])  # nothing is applied that was not saved


class SecretTests(WindowCase):
    def test_no_api_key_in_any_widget_state_or_log(self):
        self.cfg_path.write_text(f'[llm]\nbackend = "anthropic"\napi_key = "{SECRET}"\n[tts]\nspeed = 1.1\n', "utf-8")
        host, _h, _ = ui_preview.build_host(home=self.home)
        records = []

        class Grab(logging.Handler):
            def emit(self, record):
                records.append(record.getMessage() + (self.format(record) if record.exc_info else ""))

        handler = Grab(logging.DEBUG)
        logging.getLogger().addHandler(handler)
        old_level = logging.getLogger().level
        logging.getLogger().setLevel(logging.DEBUG)
        self.addCleanup(logging.getLogger().removeHandler, handler)
        self.addCleanup(logging.getLogger().setLevel, old_level)
        self.open(host)
        self.set_speed(1.3)
        self.on_ui(self.win.save)
        self.wait_banner("Applied")

        def dump():
            out = []

            def walk(w):
                for opt in ("text", "textvariable", "values"):
                    try:
                        out.append(str(w.cget(opt)))
                    except Exception:
                        pass
                try:
                    out.append(str(w.tk.globalgetvar(str(w.cget("textvariable")))))
                except Exception:
                    pass
                for c in w.winfo_children():
                    walk(c)

            walk(self.win.top)
            out += [repr(v.to_value()) for v in self.win.fields.values()]
            out += [str(self.win.base.llm), repr(self.win.base)]  # repr=False on the field
            return "\n".join(out)

        self.assertNotIn(SECRET, self.on_ui(dump))
        self.assertNotIn("llm.api_key", self.win.fields)
        changes, _ok = self.on_ui(self.win.collect)
        self.assertFalse([k for k in changes if "api_key" in k])
        self.assertFalse([r for r in records if SECRET in r])
        self.assertIn(SECRET, self.cfg_path.read_text("utf-8"))  # Save left the file as it was (the migration owns that line)


class TrayWiringTests(unittest.TestCase):
    """TrayApp.open_settings -> a real window; shutdown closes it (the `--tray --smoke` path, with a fake tray)."""

    @classmethod
    def setUpClass(cls):
        if not tk_ok():
            raise unittest.SkipTest(f"Tk cannot start here ({ui.last_error()})")

    def tearDown(self):
        sw.close_settings()
        end = time.monotonic() + 5
        while ui.call(lambda: sw._current is not None and sw._current.alive()) and time.monotonic() < end:
            time.sleep(0.05)

    def make(self, tmp):
        from parley import tray_mode
        from tests.test_tray_mode import FakeTray

        args = argparse.Namespace(config=Path(tmp) / "config.toml", restarted=False, verbose=False)
        return tray_mode.TrayApp(args, lambda a, c: None, tray_factory=FakeTray, welcome_dir=Path(tmp), smoke=True)

    def test_settings_opens_and_quit_closes_it(self):
        with tempfile.TemporaryDirectory(prefix="parley_test_settingswin_") as tmp:
            app = self.make(tmp)
            with self.assertLogs("parley.settings_window", "INFO") as logs:
                app.open_settings()
                end = time.monotonic() + 10
                while not any("window opened" in m for m in logs.output) and time.monotonic() < end:
                    time.sleep(0.05)
            self.assertTrue(any("Settings window opened" in m for m in logs.output))
            self.assertTrue(ui.call(lambda: sw._current is not None and sw._current.alive()))
            app.shutdown()  # what Quit / WM_CLOSE do
            self.assertFalse(ui._ui.alive())
            self.assertIsNone(sw._current)

    def test_second_launch_message_only_opens_the_window(self):
        from parley import tray

        # the registered message carries no data: the handler ignores wparam / lparam and calls one no-argument callback
        calls = []
        if tray.AVAILABLE:
            t = tray.Tray(on_open_settings=lambda: calls.append("open"))
            t._on_open_settings(0, 0, 123, 456)
            self.assertEqual(calls, ["open"])

    def test_menu_left_click_and_second_launch_use_the_same_entry_point(self):
        with tempfile.TemporaryDirectory(prefix="parley_test_settingswin_") as tmp:
            app = self.make(tmp)
            app._refresh_menu()
            calls = []
            with mock.patch.object(sw, "open_settings", side_effect=lambda host, page=None: calls.append(host) or True):
                for trigger in (next(i for i in app.tray.items if i.id == "settings").callback,  # menu
                                app.tray.kw["on_activate"],  # left click
                                app.tray.kw["on_open_settings"]):  # a second launch's request
                    n = len(calls)
                    app._last_settings = float("-inf")  # (the repeat guard has its own test)
                    trigger()
                    end = time.monotonic() + 5
                    while len(calls) == n and time.monotonic() < end:
                        time.sleep(0.02)
                    self.assertEqual(len(calls), n + 1)
            self.assertTrue(all(h is app.settings_host for h in calls))

    def test_one_user_action_is_one_request(self):
        with tempfile.TemporaryDirectory(prefix="parley_test_settingswin_") as tmp:
            app = self.make(tmp)
            calls = []
            with mock.patch.object(sw, "open_settings", side_effect=lambda host, page=None: calls.append(1) or True),                     self.assertLogs("parley.tray_mode", "INFO") as logs:
                app.open_settings()
                app.open_settings()  # a double click / a repeated message right behind the first
                self.assertEqual(len(calls), 1)
                self.assertEqual(sum("Settings requested" in m for m in logs.output), 1)
                app._last_settings -= tray_mode_debounce() + 0.1  # later, a new action counts
                app.open_settings()
            self.assertEqual(len(calls), 2)

    def test_restart_now_quits_and_relaunches_with_the_restarted_flag(self):
        with tempfile.TemporaryDirectory(prefix="parley_test_settingswin_") as tmp:
            app = self.make(tmp)
            launched = []
            with mock.patch("subprocess.Popen", lambda cmd, **kw: launched.append(cmd)), \
                    mock.patch("parley.autoupdate.install_root", return_value=None):
                app.restart_app()
                self.assertTrue(app.restarting)
                self.assertIn(("stop",), app.tray.calls)
                app._relaunch()
            self.assertEqual(len(launched), 1)
            self.assertIn("--restarted", launched[0])
            self.assertIn("--tray", launched[0])


class PreviewSmokeTests(unittest.TestCase):
    def test_ui_preview_opens_the_real_window_and_exits(self):
        with tempfile.TemporaryDirectory(prefix="parley_test_settingswin_") as tmp:
            env = {**os.environ, "PARLEY_HOME": str(Path(tmp) / "parley_test_ui_smoke")}
            t0 = time.monotonic()
            r = subprocess.run([sys.executable, "-m", "parley.ui_preview", "--smoke", "--theme", "dark", "--page", "voice"],
                               capture_output=True, text=True, timeout=120, env=env, cwd=ROOT)
            took = time.monotonic() - t0
        if r.returncode == 1 and "could not start" in r.stderr:
            self.skipTest("no display for Tk")
        self.assertEqual(r.returncode, 0, r.stderr[-500:])
        self.assertIn("window open after", r.stdout)
        self.assertIn("parley_test_", r.stdout)  # a throwaway home
        self.assertLess(took, 30)

    def test_ui_preview_never_uses_the_real_home(self):
        real = os.path.join(os.environ.get("LOCALAPPDATA", "x"), "parley")
        home, created = ui_preview.throwaway_home()
        try:
            self.assertTrue(home.name.startswith("parley_test_"))
            self.assertNotEqual(os.path.normcase(str(home)), os.path.normcase(real))
        finally:
            if created:
                home.rmdir()
        with mock.patch.dict(os.environ, {"PARLEY_HOME": real}):  # a real-home value is ignored, not used
            home, created = ui_preview.throwaway_home()
            try:
                self.assertTrue(created)
                self.assertTrue(home.name.startswith("parley_test_"))
            finally:
                home.rmdir()


if __name__ == "__main__":
    unittest.main()
