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

from parley import config_io, models, models_view as mv, settings_schema as S, settings_window as sw, ui, ui_preview, voices

SECRET = "TESTKEY-not-a-real-secret-0123456789"
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


class ModelsViewTests(unittest.TestCase):
    cat = models.CATALOG

    def st(self, state="installed", n=0):
        return models.Status(state, n)

    def acts(self, mid, state="installed", **kw):
        kw = {"in_use": False, "selected": False, "other_chosen": False, "dl": models.DownloadState(), **kw}
        return mv.actions(self.cat[mid], self.st(state), **kw)

    def test_sizes_and_hardware_line(self):
        self.assertEqual((mv.format_size(0), mv.format_size(74 * 1024**2), mv.format_size(int(1.5 * 1024**3))), ("-", "74 MB", "1.5 GB"))
        line = mv.hardware_line(ui_preview.stub_hardware())
        self.assertIn("6 GB video memory", line)
        self.assertIn("12 GB memory", line)
        self.assertIn("No supported graphics card", mv.hardware_line(models.Hardware(ram_mb=8192, free_disk_mb=50 * 1024)))

    def test_fit_is_word_and_symbol_never_colour_only(self):
        for level in ("good", "slow", "no"):
            text = mv.fit_text(level, "Some words")
            self.assertTrue(text.endswith("Some words") and len(text) > len("Some words"), text)
        self.assertEqual(mv.warning_text("good"), "")
        self.assertIn("slower", mv.warning_text("slow"))
        self.assertIn("at your own risk", mv.warning_text("no"))  # warns, never blocks

    def test_status_text(self):
        m = self.cat["llm-small"]
        self.assertEqual(mv.status_text(m, self.st(), True), "In use")
        self.assertEqual(mv.status_text(m, self.st(), False, "unsaved"), "Chosen (press Save)")
        self.assertEqual(mv.status_text(m, self.st(), False, "saved"), "Chosen (switching)")  # reloads live
        self.assertEqual(mv.status_text(self.cat["stt-small.en"], self.st(), False, "saved"), "Chosen (restart Parley)")
        self.assertEqual(mv.status_text(m, self.st("not_installed"), False), "Not installed")
        self.assertEqual(mv.status_text(m, self.st("downloading", m.total_bytes // 2), False), "Downloading 50%")
        self.assertTrue(mv.status_text(m, self.st("partial", 5 * 1024**2), False).startswith("Partly"))

    def test_language_model_button_rules(self):
        a = self.acts("llm-small", "not_installed")
        self.assertEqual((a.download, a.cancel, a.use, a.delete), (True, False, False, False))
        a = self.acts("llm-small")
        self.assertEqual((a.download, a.use, a.delete), (False, True, True))
        a = self.acts("llm-small", in_use=True)  # in use: no Use, no Delete, and it says why
        self.assertEqual((a.use, a.delete), (False, False))
        self.assertIn("in use", a.note)
        self.assertTrue(self.acts("llm-small", in_use=True, other_chosen=True).use)  # to switch back
        self.assertFalse(self.acts("llm-small", selected=True).delete)
        running = models.DownloadState("llm-medium", 0, 1, 10, 100, "downloading")
        a = self.acts("llm-medium", "downloading", dl=running)
        self.assertEqual((a.download, a.cancel, a.use, a.delete), (False, True, False, False))
        a = self.acts("llm-tiny", "not_installed", dl=running)  # one download at a time
        self.assertEqual((a.download, a.cancel), (False, False))
        self.assertIn("Another download", a.note)
        self.assertTrue(self.acts("llm-medium", "partial").download)  # continues where it stopped
        self.assertTrue(self.acts("llm-medium", "partial").delete)

    def test_speech_and_voice_models_have_the_same_button_rules_as_language_models(self):
        for mid in ("stt-small.en", "tts-kokoro-fp16"):
            a = self.acts(mid, "not_installed")
            self.assertEqual((a.download, a.cancel, a.use, a.delete), (True, False, False, False), mid)
            a = self.acts(mid)
            self.assertEqual((a.download, a.use, a.delete), (False, True, True), mid)
            a = self.acts(mid, in_use=True)
            self.assertEqual((a.use, a.delete), (False, False), mid)
            self.assertIn("in use", a.note)
            self.assertTrue(self.acts(mid, in_use=True, other_chosen=True).use, mid)
            self.assertFalse(self.acts(mid, selected=True).delete, mid)
        self.assertEqual(mv.LIVE_KINDS, {"llm", "stt", "tts"})

    def test_every_kind_has_a_restart_note_and_unchecked_voices_say_so(self):
        for kind, _title in mv.GROUPS:
            self.assertIn("restarting Parley" if kind != "llm" else "switches when it is ready", mv.RESTART_NOTES[kind])
        self.assertIn("not checked", self.cat["tts-kokoro-fp16"].note)
        self.assertIn("not checked", self.cat["tts-kokoro-int8"].note)
        self.assertNotIn("not checked", self.cat["tts-kokoro"].note)

    def test_models_are_listed_largest_first_within_each_group(self):
        for kind, _title in mv.GROUPS:
            got = mv.ordered(kind, self.cat)
            sizes = [m.total_bytes for m in got]
            self.assertEqual(sizes, sorted(sizes, reverse=True), kind)
            self.assertEqual({m.id for m in got}, {m.id for m in self.cat.values() if m.kind == kind})  # none dropped
        self.assertEqual([m.id for m in mv.ordered("llm", self.cat)], ["llm-medium", "llm-small", "llm-tiny"])

    def test_equal_sizes_keep_catalog_order(self):
        import dataclasses

        base = self.cat["stt-small"]
        cat = {i: dataclasses.replace(base, id=i, files=(), approx_bytes=100 * 1024**2) for i in ("stt-b", "stt-a", "stt-c")}
        cat["stt-big"] = dataclasses.replace(base, id="stt-big", files=(), approx_bytes=200 * 1024**2)
        self.assertEqual([m.id for m in mv.ordered("stt", cat)], ["stt-big", "stt-b", "stt-a", "stt-c"])

    def test_unpinned_model_cannot_be_downloaded(self):
        m = models.build_catalog({})["llm-small"]
        a = mv.actions(m, self.st("not_installed"), in_use=False, selected=False, other_chosen=False, dl=models.DownloadState())
        self.assertFalse(a.download)
        self.assertEqual(mv.status_text(m, self.st("not_installed"), False), "Not available yet")


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

    def test_save_without_an_app_but_with_restart_offers_restart_now(self):
        host = sw.Host(self.cfg_path, autostart_api=ui_preview.FakeAutostart(), list_devices=lambda k: [], restart=lambda: None)
        self.open(host)
        self.set_speed(1.25)
        self.on_ui(self.win.save)
        text, kind = self.banner()
        self.assertEqual((text, kind), ("Saved. Restart Parley to use it.", "warn"))
        self.assertTrue(self.on_ui(lambda: self.win.restart_btn.winfo_manager() != ""))

    def test_failed_restart_reenables_the_button_and_says_so(self):
        host, _h, _ = ui_preview.build_host("restart-needed", home=self.home)

        def boom():
            raise OSError("cannot start")

        host._restart = boom
        self.open(host)
        with self.assertLogs("parley.settings_window", "ERROR"):
            self.on_ui(self.win._restart_now)
            self.wait_banner("Parley couldn't restart")
        self.assertEqual(self.banner()[1], "error")
        self.assertFalse(self.on_ui(lambda: self.win.restart_btn.instate(["disabled"])))
        self.assertTrue(self.on_ui(lambda: self.win.restart_btn.winfo_manager() != ""))

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


class ModelsPageTests(WindowCase):
    """The Models page against ui_preview.StubModels (fake files, simulated download) on a fake computer."""

    def open_models(self, host=None):
        self.open(host)
        self.on_ui(lambda: self.win.show_page("models"))
        end = time.monotonic() + 5
        while time.monotonic() < end:
            if self.on_ui(lambda: self.win.model_tree is not None and self.win.model_tree.exists("llm-small")
                          and self.win.hw is not None):
                return
            time.sleep(0.05)
        self.fail("the models page never filled")

    def host_with(self, models_state=None):
        host, _h, _ = ui_preview.build_host(home=self.home, models_state=models_state)
        return host

    def row(self, mid):
        return self.on_ui(lambda: (self.win.model_tree.item(mid, "text"), *self.win.model_tree.item(mid, "values")))

    def pick(self, mid):
        self.on_ui(lambda: self.win.model_tree.selection_set(mid))
        time.sleep(0.1)

    def btns(self):
        return self.on_ui(lambda: {k: not b.instate(["disabled"]) for k, b in self.win.model_btns.items()})

    def note(self):
        return self.on_ui(lambda: self.win.model_note.cget("text"))

    def msg(self):
        return self.on_ui(lambda: self.win.model_msg.cget("text"))

    def wait_row(self, mid, status, timeout=4):
        end = time.monotonic() + timeout
        while self.row(mid)[2] != status and time.monotonic() < end:
            time.sleep(0.05)
        self.assertEqual(self.row(mid)[2], status)

    def test_every_catalog_model_is_listed_in_its_group_and_nothing_is_filtered(self):
        self.open_models(self.host_with())
        for kind, _title in mv.GROUPS:
            kids = self.on_ui(lambda k=kind: list(self.win.model_tree.get_children(f"group:{k}")))
            self.assertEqual([i for i in kids if i != mv.OWN_ID], [m.id for m in mv.ordered(kind, models.CATALOG)])  # largest first
        self.assertEqual(self.row("llm-tiny")[1:3], ("1.2 GB", "In use"))
        self.assertIn("Too big", self.row("llm-medium")[3])  # listed with the warning, not hidden
        self.assertTrue(self.row("llm-small")[0].endswith("recommended"))
        self.assertIn("12 GB memory", self.on_ui(lambda: self.win.hw_text.get()))

    def test_in_use_model_cannot_be_deleted_and_the_row_explains(self):
        self.open_models(self.host_with("downloading"))
        self.pick("llm-tiny")
        self.assertEqual(self.btns(), {"download": False, "cancel": False, "use": False, "delete": False})
        self.assertIn("in use", self.note())
        self.pick("llm-medium")  # the one downloading in this preview
        self.assertEqual(self.btns(), {"download": False, "cancel": True, "use": False, "delete": False})
        detail = self.on_ui(lambda: self.win.model_detail.cget("text"))
        self.assertIn("Too big for this computer", detail)
        self.assertIn("at your own risk", detail)

    def test_speech_and_voice_rows_can_be_downloaded_chosen_and_deleted(self):
        host = self.host_with()
        self.open_models(host)
        self.pick("stt-tiny.en")  # not installed in the stub
        self.assertEqual(self.btns(), {"download": True, "cancel": False, "use": False, "delete": False})
        self.pick("stt-base.en")  # installed, not running
        self.assertEqual(self.btns(), {"download": False, "cancel": False, "use": True, "delete": True})
        self.assertIn("speech model changes after restarting", self.note())
        self.pick("stt-small.en")  # running
        self.assertEqual(self.btns(), {"download": False, "cancel": False, "use": False, "delete": False})
        self.pick("tts-kokoro")
        self.assertFalse(self.btns()["delete"])  # the standard voice is chosen and running
        self.pick("tts-kokoro-fp16")
        self.assertEqual(self.btns(), {"download": False, "cancel": False, "use": True, "delete": True})
        self.assertIn("voice model changes after restarting", self.note())
        self.assertIn("Sound quality not checked yet", self.on_ui(lambda: self.win.model_detail.cget("text")))
        self.pick("tts-kokoro-int8")
        self.assertTrue(self.btns()["download"])

    def test_use_on_speech_sets_stt_model_and_save_writes_only_that_key(self):
        self.open_models(self.host_with())
        self.pick("stt-base.en")
        self.on_ui(self.win._model_use)
        self.assertEqual(self.row("stt-base.en")[2], "Chosen (press Save)")
        self.assertNotIn("base.en", self.cfg_path.read_text("utf-8"))  # nothing is written before Save
        changes, ok = self.on_ui(self.win.collect)
        self.assertTrue(ok)
        self.assertEqual(changes, {"stt.model": "base.en"})
        self.on_ui(self.win.save)
        self.wait_banner("Saved. 1 setting needs a restart")
        self.assertIn('model = "base.en"', self.cfg_path.read_text("utf-8"))
        self.assertIn("# my favourite", self.cfg_path.read_text("utf-8"))
        self.assertEqual(self.row("stt-base.en")[2], "Chosen (restart Parley)")
        self.assertEqual(self.row("stt-small.en")[2], "In use")  # still what is running

    def test_use_on_voice_maps_the_catalog_id_to_the_variant(self):
        self.open_models(self.host_with())
        self.pick("tts-kokoro-fp16")
        self.on_ui(self.win._model_use)
        changes, ok = self.on_ui(self.win.collect)
        self.assertTrue(ok)
        self.assertEqual(changes, {"tts.model": "fp16"})
        self.assertEqual(self.row("tts-kokoro-fp16")[2], "Chosen (press Save)")
        self.assertTrue(self.row("tts-kokoro-fp16")[0].startswith("Kokoro (half precision)"))
        self.assertTrue(self.row("tts-kokoro-fp16")[0].endswith("recommended"))  # the chosen file carries the mark (recommended_ids)
        self.on_ui(self.win.save)
        self.wait_banner("Saved. 1 setting needs a restart")
        self.assertIn('model = "fp16"', self.cfg_path.read_text("utf-8"))
        self.assertEqual(self.row("tts-kokoro-fp16")[2], "Chosen (restart Parley)")
        self.assertEqual(self.row("tts-kokoro")[2], "In use")
        self.pick("tts-kokoro-fp16")  # chosen already: nothing more to do, and it cannot be deleted
        self.assertEqual(self.btns(), {"download": False, "cancel": False, "use": False, "delete": False})

    def test_choosing_in_each_section_keeps_the_other_choices(self):
        self.open_models(self.host_with())
        for mid in ("llm-small", "stt-base.en", "tts-kokoro-fp16"):
            self.pick(mid)
            self.on_ui(self.win._model_use)
        changes, _ok = self.on_ui(self.win.collect)
        self.assertEqual(changes, {"llm.preset": "small", "stt.model": "base.en", "tts.model": "fp16"})

    def test_use_sets_pending_values_and_save_writes_only_those_keys(self):
        self.open_models(self.host_with())
        self.pick("llm-small")
        self.assertTrue(self.btns()["use"])
        self.assertIn("switches when it is ready", self.note())
        self.on_ui(self.win._model_use)
        self.assertEqual(self.row("llm-small")[2], "Chosen (press Save)")
        self.assertNotIn("llm", self.cfg_path.read_text("utf-8"))  # nothing is written before Save
        changes, ok = self.on_ui(self.win.collect)
        self.assertTrue(ok)
        self.assertEqual(changes, {"llm.preset": "small"})  # the empty repo / file / path already match config.toml
        self.on_ui(self.win.save)
        self.wait_banner("Saved. Loading the language model")  # llm.preset reloads live now (the stub takes 1.5 s)
        out = self.cfg_path.read_text("utf-8")
        self.assertIn('preset = "small"', out)
        self.assertIn("# my favourite", out)
        self.assertEqual(self.row("llm-small")[2], "Chosen (switching)")
        self.assertEqual(self.row("llm-tiny")[2], "In use")  # still what is running

    def test_use_warns_that_it_replaces_a_custom_model_only_when_there_is_one(self):
        self.open_models(self.host_with())
        self.pick("llm-small")
        self.assertNotIn(mv.OWN_REPLACE_NOTE, self.note())  # no custom model: no warning
        self.on_ui(self.win.close)
        self.cfg_path.write_text('[llm]\nmodel_repo = "someone/their-model-GGUF"\nmodel_file = "x.gguf"\n', "utf-8")
        host = self.host_with()
        with mock.patch.object(models, "own_llm_file", return_value=models.OwnFile("Your own file", self.home / "x.gguf", 5)):
            self.open_models(host)
            self.pick("llm-small")
            self.assertIn(mv.OWN_REPLACE_NOTE, self.note())
            self.assertIn("switches when it is ready", self.note())

    def test_use_replaces_a_custom_model(self):
        self.cfg_path.write_text('[llm]\nmodel_repo = "someone/their-model-GGUF"\nmodel_file = "x.gguf"\n', "utf-8")
        self.open_models(self.host_with())
        self.pick("llm-small")
        self.on_ui(self.win._model_use)
        changes, _ok = self.on_ui(self.win.collect)
        self.assertEqual(changes, {"llm.preset": "small", "llm.model_repo": "", "llm.model_file": ""})

    def test_download_progress_cancel_and_one_at_a_time(self):
        host = self.host_with("downloading")
        self.open_models(host)
        self.pick("llm-medium")
        time.sleep(0.6)
        self.assertTrue(self.on_ui(lambda: self.win.model_progress.winfo_manager() != ""))
        self.assertRegex(self.on_ui(lambda: self.win.model_bar_text.cget("text")), r"of 5\.3 GB \(\d+%\)")
        self.assertGreaterEqual(self.on_ui(lambda: float(self.win.model_bar.cget("value"))), 43)
        host.models._installed.discard("llm-small")  # a model that could be downloaded, if nothing else was running
        self.on_ui(self.win._models_refresh)
        time.sleep(0.4)
        self.pick("llm-small")
        self.assertFalse(self.btns()["download"])
        self.assertIn("Another download", self.note())
        self.pick("llm-medium")
        self.on_ui(self.win.model_btns["cancel"].invoke)
        end = time.monotonic() + 3
        while "cancelled" not in self.msg() and time.monotonic() < end:
            time.sleep(0.05)
        self.assertIn("cancelled", self.msg())
        end = time.monotonic() + 3
        while not self.row("llm-medium")[2].startswith("Partly") and time.monotonic() < end:
            time.sleep(0.05)
        self.assertTrue(self.row("llm-medium")[2].startswith("Partly"))
        self.assertEqual(self.on_ui(lambda: self.win.model_progress.winfo_manager()), "")

    def test_a_failed_download_is_reported_in_plain_words(self):
        host = self.host_with()
        self.open_models(host)
        self.pick("llm-medium")
        self.on_ui(self.win._model_download)
        st = host.models.state
        self.assertEqual(st.phase, "downloading")
        with host.models._lock:  # the worker reports a failed safety check
            host.models._dl = models.DownloadState("llm-medium", 0, 1, 5, 100, "error", "The Qwen3.5 9B (medium) download failed its safety check, so Parley discarded it.")
        end = time.monotonic() + 3
        while "safety check" not in self.msg() and time.monotonic() < end:
            time.sleep(0.05)
        self.assertIn("safety check", self.msg())

    def test_download_click_passes_the_catalog_id_only_and_a_second_is_refused(self):
        host = self.host_with()
        calls = []
        real = host.models.download
        host.models.download = lambda mid: calls.append(mid) or real(mid)
        self.open_models(host)
        self.pick("llm-medium")
        self.assertTrue(self.btns()["download"])
        self.on_ui(self.win._model_download)
        self.assertEqual(calls, ["llm-medium"])
        time.sleep(0.6)
        self.assertTrue(self.row("llm-medium")[2].startswith("Downloading"))
        self.assertFalse(self.btns()["download"])
        self.on_ui(self.win._model_download)  # forced past the disabled button: the manager says no
        self.assertIn("already downloading", self.msg())

    def test_delete_asks_first_and_frees_the_space(self):
        host = self.host_with()
        self.open_models(host)
        self.pick("llm-small")
        with mock.patch.object(sw.messagebox, "askyesno", return_value=False) as ask:
            self.on_ui(self.win._model_delete)
        self.assertIn("Qwen3.5 4B", ask.call_args.args[1])
        self.assertIn("llm-small", host.models._installed)  # declined: nothing deleted
        with mock.patch.object(sw.messagebox, "askyesno", return_value=True):
            self.on_ui(self.win._model_delete)
        self.assertNotIn("llm-small", host.models._installed)
        self.assertIn("freed", self.msg())
        self.wait_row("llm-small", "Not installed")

    def test_without_a_manager_the_page_says_so_and_nothing_breaks(self):
        host = sw.Host(self.cfg_path, autostart_api=ui_preview.FakeAutostart(), list_devices=lambda k: [])
        self.open(host)
        self.on_ui(lambda: self.win.show_page("models"))
        time.sleep(0.5)
        self.assertIsNone(self.on_ui(lambda: self.win.model_tree))

    def test_closing_mid_download_leaves_it_running_and_touches_no_tk(self):
        host = self.host_with("downloading")
        self.open_models(host)
        self.on_ui(self.win.close)
        time.sleep(0.8)  # polls and a scan may still be queued: none may raise
        self.assertEqual(host.models.state.phase, "downloading")

    def test_a_failing_scan_gives_a_plain_message(self):
        host = self.host_with()

        def boom(lang):
            raise OSError("no probe")

        host._detect_hardware = boom
        with self.assertLogs("parley.settings_window", "ERROR"):
            self.open(host)
            self.on_ui(lambda: self.win.show_page("models"))
            end = time.monotonic() + 5
            while "couldn't read" not in self.msg() and time.monotonic() < end:
                time.sleep(0.05)
        self.assertIn("couldn't read", self.msg())


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


KEY = "sk-ant-" "api03-TESTKEY0123456789abcdef"
KEY2 = "TESTKEY-not-a-real-secret-ZZZZ0123456789"


def widget_dump(win) -> str:
    """Every text, value list and variable of every widget, the field values and the config objects (UI thread)."""
    out: list[str] = []

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
        try:
            out.append(str(w.get("1.0", "end")))  # a Text widget
        except Exception:
            try:
                out.append(str(w.get()))  # an Entry: what is typed in it right now
            except Exception:
                pass
        for c in w.winfo_children():
            walk(c)

    walk(win.top)
    out += [repr(f.to_value()) for f in win.fields.values()]
    out += [str(win.base.llm), repr(win.base), win.banner.cget("text")]
    return "\n".join(out)


class LlmPageTests(WindowCase):
    """The LLM page: backend fields, API keys through the credential store only, the cloud notice, live reload results."""

    def setUp(self):
        super().setUp()
        self.records: list[str] = []

        class Grab(logging.Handler):
            def emit(inner, record):
                self.records.append(record.getMessage() + (logging.Formatter().formatException(record.exc_info) if record.exc_info else ""))

        handler = Grab(logging.DEBUG)
        root = logging.getLogger()
        root.addHandler(handler)
        old = root.level
        root.setLevel(logging.DEBUG)
        self.addCleanup(lambda: (root.removeHandler(handler), root.setLevel(old)))
        for var in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY"):
            patcher = mock.patch.dict(os.environ)
            patcher.start()
            self.addCleanup(patcher.stop)
            os.environ.pop(var, None)

    def open_llm(self, backend="anthropic", extra="", state=None, host=None):
        self.cfg_path.write_text(f'# my notes\n[tts]\nspeed = 1.1  # fast\n[llm]\nbackend = "{backend}"\n{extra}', "utf-8")
        self.host, _h, _ = ui_preview.build_host(state, home=self.home, page="llm")
        self.app = self.host.app
        self.keys = self.host.keys
        self.win = ui.call(lambda: sw.SettingsWindow(ui.root(), host or self.host, "llm"))
        return self.win

    def key_row(self, backend="anthropic"):
        return self.on_ui(lambda: self.win.keys[backend])

    def type_key(self, text, backend="anthropic"):
        def go():
            e = self.win.keys[backend].entry
            e.delete(0, "end")
            e.insert(0, text)
        self.on_ui(go)

    def key_state(self, backend="anthropic"):
        def go():
            k = self.win.keys[backend]
            return {"status": k.status.cget("text"), "msg": k.message.cget("text"), "entry": k.entry.get(), "save": k.save.cget("text"),
                    "remove": not k.remove.instate(["disabled"]), "test": not k.test.instate(["disabled"]), "show": k.entry.cget("show")}
        return self.on_ui(go)

    def wait_msg(self, startswith, backend="anthropic", timeout=5):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if self.key_state(backend)["msg"].startswith(startswith):
                return
            time.sleep(0.05)
        self.fail(f"key message never started with {startswith!r}: {self.key_state(backend)['msg']!r}")

    def assert_no_key_anywhere(self, *keys):
        dump = self.on_ui(lambda: widget_dump(self.win))
        for key in keys:
            self.assertNotIn(key, dump)
            self.assertFalse([r for r in self.records if key in r], "a log record holds the key")
            self.assertNotIn(key, self.cfg_path.read_text("utf-8"))
            for f in self.home.rglob("*"):
                if f.is_file() and f.stat().st_size < 1_000_000:
                    self.assertNotIn(key.encode(), f.read_bytes(), f.name)

    # -- the page itself
    def test_page_is_in_the_navigation_after_models(self):
        self.open_llm()
        self.assertEqual([k for k, _t in sw.PAGES][-2:], ["models", "llm"])
        self.assertEqual(self.on_ui(lambda: self.win.page), "llm")

    def test_fields_come_from_config_and_the_key_is_not_one_of_them(self):
        self.open_llm("openai", 'model = "gpt-test"\nbase_url = "https://proxy.example.com/v1"\n')
        values = self.on_ui(lambda: {p: f.to_value() for p, f in self.win.fields.items()})
        self.assertEqual(values["llm.backend"], "openai")
        self.assertEqual(values["llm.model"], "gpt-test")
        self.assertEqual(values["llm.base_url"], "https://proxy.example.com/v1")
        self.assertEqual(values["llm.system_prompt"], sw.config.DEFAULT_SYSTEM_PROMPT)
        self.assertNotIn("llm.api_key", values)
        changes, ok = self.on_ui(self.win.collect)
        self.assertEqual((changes, ok), ({}, True))

    def test_every_listed_backend_is_a_real_backend_and_nothing_is_duplicated_from_other_pages(self):
        self.assertEqual(set(sw.BACKEND_LABELS), set(S.BACKENDS))
        self.assertTrue(set(sw.LLM_BACKENDS) <= set(S.BY_PATH))
        self.open_llm()
        on_llm = {p for p in self.on_ui(lambda: set(self.win.fields)) if p.startswith(("llm.", "tools."))}
        self.assertFalse(on_llm & set((*sw.GENERAL_FIELDS, *sw.VOICE_FIELDS)))  # nothing is on two pages
        self.assertFalse([p for p in (*sw.GENERAL_FIELDS, *sw.VOICE_FIELDS) if p.startswith(("llm.", "tools."))])
        self.assertIn("tools.enabled", on_llm)

    def test_settings_of_other_backends_are_hidden_and_not_checked(self):
        self.open_llm("anthropic")

        def go():
            self.win.fields["llm.num_ctx"].widgets[0].delete(0, "end")  # a builtin / ollama setting
            self.win.fields["llm.num_ctx"].widgets[0].insert(0, "abc")
            return (self.win.fields["llm.num_ctx"].row.winfo_manager() == "", self.win.collect())
        hidden, (changes, ok) = self.on_ui(go)
        self.assertTrue(hidden)
        self.assertTrue(ok)
        self.assertNotIn("llm.num_ctx", changes)

        def switch():
            self.win.backend_var.set(sw.BACKEND_LABELS["builtin"])
            self.win._backend_changed()
            return self.win.collect(), dict(self.win.errors)
        (changes, ok), errors = self.on_ui(switch)
        self.assertFalse(ok)  # now it is visible, so it is checked
        self.assertIn("llm.num_ctx", errors)
        self.assertTrue(self.save_disabled())

    def test_backend_rows_follow_the_selection(self):
        self.open_llm("builtin")

        def shown():
            return {"key_anthropic": self.win.keys["anthropic"].frame.winfo_manager() != "",
                    "key_openai": self.win.keys["openai"].frame.winfo_manager() != "",
                    "model": self.win.fields["llm.model"].row.winfo_manager() != "",
                    "effort": self.win.fields["llm.effort"].row.winfo_manager() != "",
                    "base_url": self.win.fields["llm.base_url"].row.winfo_manager() != ""}

        def pick(backend):
            self.win.backend_var.set(sw.BACKEND_LABELS[backend])
            self.win._backend_changed()
            return shown()
        self.assertEqual(self.on_ui(shown), {"key_anthropic": False, "key_openai": False, "model": False, "effort": False, "base_url": False})
        self.assertEqual(self.on_ui(lambda: pick("anthropic")), {"key_anthropic": True, "key_openai": False, "model": True, "effort": True, "base_url": False})
        self.assertEqual(self.on_ui(lambda: pick("openai")), {"key_anthropic": False, "key_openai": True, "model": True, "effort": False, "base_url": True})
        self.assertEqual(self.on_ui(lambda: pick("server")), {"key_anthropic": False, "key_openai": False, "model": True, "effort": False, "base_url": True})

    def test_save_writes_only_the_changed_keys_keeps_comments_and_reloads_live(self):
        self.open_llm("anthropic")

        def go():
            self.win.fields["llm.model"].widgets[0].insert(0, "claude-test")
            self.win.fields["llm.history_turns"].widgets[0].delete(0, "end")
            self.win.fields["llm.history_turns"].widgets[0].insert(0, "7")
        self.on_ui(go)
        changes, ok = self.on_ui(self.win.collect)
        self.assertEqual((changes, ok), ({"llm.model": "claude-test", "llm.history_turns": 7}, True))
        self.on_ui(self.win.save)
        self.wait_banner("Saved. Loading the language model")
        out = self.cfg_path.read_text("utf-8")
        self.assertIn('model = "claude-test"', out)
        self.assertIn("history_turns = 7", out)
        self.assertIn("# my notes", out)
        self.assertIn("# fast", out)
        self.assertEqual(self.wait_banner("Saved. The new language model is ready", timeout=6)[1], "ok")
        self.assertEqual(self.app.llm_state[0], "done")

    def test_a_failed_reload_is_a_plain_sentence_and_the_old_model_stays(self):
        self.open_llm("anthropic", state="llm-error")
        self.on_ui(lambda: self.win.fields["llm.model"].widgets[0].insert(0, "claude-test"))
        self.on_ui(self.win.save)
        text, kind = self.wait_banner("Saved, but the new language model could not start", timeout=6)
        self.assertEqual(kind, "error")
        self.assertIn("still using the previous one", text)
        self.assertTrue(self.on_ui(lambda: self.win.restart_btn.winfo_manager() != ""))

    def test_a_busy_app_says_the_change_waits_for_the_answer(self):
        self.open_llm("anthropic")
        self.app.apply_config = lambda new: ([], [])
        self.app.deferred = ["llm.model"]
        self.on_ui(lambda: self.win.fields["llm.model"].widgets[0].insert(0, "claude-test"))
        self.on_ui(self.win.save)
        self.wait_banner("Saved. 1 setting will apply after this answer")

    # -- the cloud notice and the address warning
    def test_cloud_backends_show_a_one_line_notice_that_text_goes_to_the_provider(self):
        self.open_llm("builtin")
        shown = lambda: self.on_ui(lambda: (self.win.llm_notice.cget("text"), self.win.llm_notice.winfo_manager() != ""))  # noqa: E731
        self.assertEqual(shown(), ("", False))
        for backend, provider in (("anthropic", "Anthropic"), ("openai", "OpenAI")):
            self.on_ui(lambda b=backend: (self.win.backend_var.set(sw.BACKEND_LABELS[b]), self.win._backend_changed()))
            text, visible = shown()
            self.assertTrue(visible)
            self.assertIn(f"sent to {provider}", text)
            self.assertIn("tools", text)
            self.assertIn("turned into text on this computer", text)
        for backend in ("ollama", "server", "builtin"):
            self.on_ui(lambda b=backend: (self.win.backend_var.set(sw.BACKEND_LABELS[b]), self.win._backend_changed()))
            self.assertEqual(shown(), ("", False))

    def test_local_backends_show_a_muted_example_and_the_page_shows_no_addresses(self):
        import re
        self.open_llm("builtin")

        def pick(b):
            self.win.backend_var.set(sw.BACKEND_LABELS[b])
            self.win._backend_changed()
            return self.win.llm_example.cget("text"), self.win.llm_example.winfo_manager() != ""
        self.assertIn("Qwen3.5-9B-GGUF", self.on_ui(lambda: pick("builtin"))[0])
        self.assertEqual(self.on_ui(lambda: pick("ollama")), ("Example model name: qwen3:4b", True))
        self.assertTrue(self.on_ui(lambda: pick("server"))[1])
        self.assertEqual(self.on_ui(lambda: pick("openai")), ("", False))
        self.assertEqual(self.on_ui(lambda: pick("anthropic")), ("", False))
        self.assertNotIn("llm.model", self.on_ui(lambda: self.win.collect())[0])  # examples are never values
        for b in ("builtin", "ollama", "server", "openai"):
            self.on_ui(lambda b=b: pick(b))
            text = self.on_ui(lambda: widget_dump(self.win))
            self.assertFalse(re.search(r"\d+\.\d+\.\d+\.\d+|localhost|https?://|:\d{4,5}", text), b)

    def test_http_address_warning(self):
        self.open_llm("openai")

        def url(text, backend=None):
            def go():
                if backend:
                    self.win.backend_var.set(sw.BACKEND_LABELS[backend])
                    self.win._backend_changed()
                e = self.win.fields["llm.base_url"].widgets[0]
                e.delete(0, "end")
                e.insert(0, text)
                return self.win.url_warning.cget("text"), self.win.url_warning.winfo_manager() != ""
            return self.on_ui(go)
        text, visible = url("http://api.openai.com/v1")  # the old check stayed silent for the provider's own host name
        self.assertTrue(visible)
        self.assertIn("unencrypted", text)
        self.assertEqual(url("https://api.openai.com/v1"), ("", False))
        self.assertEqual(url(""), ("", False))
        text, visible = url("https://proxy.example.com/v1")
        self.assertTrue(visible)
        self.assertIn("another address", text)
        self.assertNotIn("proxy.example.com", text)  # the page never spells out the host
        self.assertEqual(url("http://127.0.0.1:8080/v1", "server"), ("", False))
        self.assertTrue(url("http://192.168.1.20:11434", "ollama")[1])
        self.assertEqual(url("https://my.server.example/v1", "server"), ("", False))  # a custom host is the point of "server"

    # -- keys
    def test_not_saved_status_and_buttons(self):
        self.open_llm("anthropic")
        st = self.key_state()
        self.assertEqual(st["status"], "No key saved.")
        self.assertEqual((st["save"], st["remove"], st["test"]), ("Save key", False, False))
        self.assertEqual(st["show"], "*")  # masked
        self.assertEqual(st["entry"], "")

    def test_saved_status_is_shown_without_the_key_and_the_entry_is_never_prefilled(self):
        self.open_llm("anthropic", state="key-saved")
        st = self.key_state()
        self.assertEqual(st["status"], "Saved in the Windows credential store.")
        self.assertEqual((st["save"], st["remove"], st["test"], st["entry"]), ("Replace key", True, True, ""))
        os.environ["ANTHROPIC_API_KEY"] = KEY2  # not even an environment key is put into the field
        self.on_ui(self.win.close)
        self.open_llm("anthropic", state="key-saved")
        st = self.key_state()
        self.assertIn("ANTHROPIC_API_KEY in the environment is ignored", st["status"])
        self.assertEqual(st["entry"], "")
        self.assert_no_key_anywhere(KEY2, "PREVIEW-not-a-real-key-0123456789")

    def test_environment_key_is_reported_by_name_only(self):
        os.environ["OPENAI_API_KEY"] = KEY2
        self.open_llm("openai")
        st = self.key_state("openai")
        self.assertIn("OPENAI_API_KEY from the environment", st["status"])
        self.assertTrue(st["test"])
        self.assertFalse(st["remove"])
        self.assert_no_key_anywhere(KEY2)

    def test_save_key_goes_to_the_credential_store_only_and_clears_the_entry(self):
        self.open_llm("builtin")  # not the running backend: nothing to reload
        before = self.cfg_path.read_bytes()
        self.on_ui(lambda: (self.win.backend_var.set(sw.BACKEND_LABELS["anthropic"]), self.win._backend_changed()))
        self.type_key(KEY)
        self.on_ui(lambda: self.win._key_save("anthropic"))
        st = self.key_state()
        self.assertEqual(st["entry"], "")  # cleared
        self.assertEqual(st["msg"], "Key saved.")
        self.assertEqual((st["save"], st["remove"], st["test"]), ("Replace key", True, True))
        self.assertEqual(self.keys.get("anthropic"), KEY)
        self.assertEqual(self.cfg_path.read_bytes(), before)  # config.toml never saw it
        self.assertEqual(self.app.keys_changed, [])
        self.assert_no_key_anywhere(KEY)

    def test_save_key_for_the_running_backend_rebuilds_the_chat(self):
        self.open_llm("anthropic")
        self.app.cfg.llm.backend = "anthropic"
        self.type_key(KEY)
        self.on_ui(lambda: self.win._key_save("anthropic"))
        self.wait_msg("Key saved. Loading it now")
        end = time.monotonic() + 3
        while not self.app.keys_changed and time.monotonic() < end:
            time.sleep(0.02)
        self.assertEqual(self.app.keys_changed, ["anthropic"])
        self.wait_banner("Saved. The new language model is ready", timeout=6)
        self.assert_no_key_anywhere(KEY)

    def test_replace_overwrites_and_a_failed_save_keeps_the_old_key(self):
        self.open_llm("anthropic", state="key-saved")
        old = self.keys.get("anthropic")
        self.keys.backend.fail = True
        self.type_key(KEY2)
        self.on_ui(lambda: self.win._key_save("anthropic"))
        st = self.key_state()
        self.assertEqual(st["entry"], "")  # cleared even when it failed
        self.assertTrue(st["msg"].startswith("The Windows credential store failed"))
        self.assertEqual(self.keys.get("anthropic"), old)
        self.keys.backend.fail = False
        self.type_key(KEY2)
        self.on_ui(lambda: self.win._key_save("anthropic"))
        self.assertEqual(self.keys.get("anthropic"), KEY2)
        self.assert_no_key_anywhere(KEY2, old)

    def test_bad_input_is_refused_without_echoing_it(self):
        self.open_llm("anthropic")
        for text, expect in (("", "Type or paste the key first."), ("has a space inside", "doesn't look like an API key"),
                             ("café-key-12345", "doesn't look like an API key")):
            self.type_key(text)
            self.on_ui(lambda: self.win._key_save("anthropic"))
            st = self.key_state()
            self.assertIn(expect, st["msg"])
            self.assertEqual(st["entry"], "")
            self.assertNotIn("inside", st["msg"].replace("doesn't look like an API key", ""))
        self.assertEqual(self.keys.backend.items, {})

    def test_remove_asks_first_and_clears_the_stored_key(self):
        self.open_llm("anthropic", state="key-saved")
        with mock.patch.object(sw.messagebox, "askyesno", return_value=False) as ask:
            self.on_ui(lambda: self.win._key_remove("anthropic"))
        ask.assert_called_once()
        self.assertTrue(self.keys.exists("anthropic"))
        with mock.patch.object(sw.messagebox, "askyesno", return_value=True):
            self.on_ui(lambda: self.win._key_remove("anthropic"))
        self.assertFalse(self.keys.exists("anthropic"))
        st = self.key_state()
        self.assertEqual((st["status"], st["save"], st["remove"], st["test"]), ("No key saved.", "Save key", False, False))
        self.assertTrue(st["msg"].startswith("Key removed."))

    def test_removing_the_key_of_the_running_backend_says_the_session_keeps_it(self):
        self.open_llm("anthropic", state="key-saved")
        self.app.cfg.llm.backend = "anthropic"
        with mock.patch.object(sw.messagebox, "askyesno", return_value=True):
            self.on_ui(lambda: self.win._key_remove("anthropic"))
        self.assertIn("until you restart", self.key_state()["msg"])

    def test_test_key_is_user_triggered_and_shows_one_sentence(self):
        calls = []

        def check(backend, key):
            calls.append((backend, key))
            return "The provider accepted this key."
        self.open_llm("anthropic", state="key-saved")
        self.host.key_check = check
        time.sleep(0.3)
        self.assertEqual(calls, [])  # nothing is sent until the button is pressed
        self.on_ui(lambda: self.win._key_test("anthropic"))
        self.wait_msg("The provider accepted this key.")
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][0], "anthropic")
        self.assertEqual(calls[0][1], "PREVIEW-not-a-real-key-0123456789")
        self.assert_no_key_anywhere("PREVIEW-not-a-real-key-0123456789")

    def test_a_crashing_key_check_gives_a_plain_sentence(self):
        def boom(backend, key):
            raise RuntimeError(f"failed with {key}")
        self.open_llm("anthropic", state="key-saved")
        self.host.key_check = boom
        self.on_ui(lambda: self.win._key_test("anthropic"))
        self.wait_msg("The key could not be tested.")
        self.assert_no_key_anywhere("PREVIEW-not-a-real-key-0123456789")

    def test_without_a_credential_store_the_page_says_so_and_nothing_can_be_saved(self):
        self.open_llm("anthropic")
        self.host.keys = None
        self.on_ui(lambda: self.win._key_refresh("anthropic"))
        st = self.key_state()
        self.assertIn("needs the Windows credential store", st["status"])
        self.assertIn("ANTHROPIC_API_KEY", st["status"])
        self.assertEqual((st["remove"], st["test"]), (False, False))
        self.assertTrue(self.on_ui(lambda: self.win.keys["anthropic"].save.instate(["disabled"])))

    def test_save_of_the_page_writes_no_key_even_with_one_typed_and_a_legacy_key_in_the_file(self):
        self.open_llm("anthropic", f'api_key = "{KEY2}"\n')
        st = self.key_state()
        self.assertIn("still in config.toml", st["status"])
        self.type_key(KEY)  # typed, not saved: Save must ignore it
        self.on_ui(lambda: self.win.fields["llm.model"].widgets[0].insert(0, "claude-test"))
        changes, _ok = self.on_ui(self.win.collect)
        self.assertEqual(changes, {"llm.model": "claude-test"})
        self.on_ui(self.win.save)
        self.wait_banner("Saved. Loading")
        out = self.cfg_path.read_text("utf-8")
        self.assertNotIn(KEY, out)
        self.assertIn(KEY2, out)  # Save left the legacy line to the startup migration
        self.assertIn("# my notes", out)
        self.assertFalse(self.keys.exists("anthropic"))  # and did not route the typed key anywhere
        self.assertFalse([r for r in self.records if KEY in r or KEY2 in r])

    def test_no_key_in_any_widget_log_or_file_through_a_whole_session(self):
        self.open_llm("anthropic")
        self.app.cfg.llm.backend = "anthropic"
        self.type_key(KEY)
        self.on_ui(lambda: self.win._key_save("anthropic"))
        self.wait_msg("Key saved. Loading it now")
        self.host.key_check = lambda b, k: "The provider rejected this key."
        self.on_ui(lambda: self.win._key_test("anthropic"))
        self.wait_msg("The provider rejected this key.")
        self.on_ui(lambda: self.win.fields["llm.model"].widgets[0].insert(0, "claude-test"))
        self.on_ui(self.win.save)
        self.wait_banner("Saved")
        with mock.patch.object(sw.messagebox, "askyesno", return_value=True):
            self.on_ui(lambda: self.win._key_remove("anthropic"))
        self.assert_no_key_anywhere(KEY)
        self.assertNotIn(KEY, "\n".join(self.records))


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

    def test_missing_tkinter_keeps_the_tray_running(self):
        import builtins

        real = builtins.__import__

        def fake(name, globals=None, locals=None, fromlist=(), level=0):
            if level == 1 and "settings_window" in (fromlist or ()):
                raise ImportError("No module named tkinter")
            return real(name, globals, locals, fromlist, level)

        with tempfile.TemporaryDirectory(prefix="parley_test_settingswin_") as tmp:
            with mock.patch.object(builtins, "__import__", fake), self.assertLogs("parley.tray_mode", "WARNING"):
                app = self.make(tmp)
            self.assertIsNone(app.settings_host)
            app.open_settings()
            balloons = [c for c in app.tray.calls if c[0] == "balloon"]
            self.assertEqual(len(balloons), 1)
            self.assertIn("Settings could not start", balloons[0][2])

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

    def test_ui_preview_models_page_opens(self):
        with tempfile.TemporaryDirectory(prefix="parley_test_settingswin_") as tmp:
            env = {**os.environ, "PARLEY_HOME": str(Path(tmp) / "parley_test_ui_smoke")}
            r = subprocess.run([sys.executable, "-m", "parley.ui_preview", "--smoke", "--page", "models", "--state", "download-error"],
                               capture_output=True, text=True, timeout=120, env=env, cwd=ROOT)
        if r.returncode == 1 and "could not start" in r.stderr:
            self.skipTest("no display for Tk")
        self.assertEqual(r.returncode, 0, r.stderr[-500:])

    def test_ui_preview_llm_page_opens_in_each_state(self):
        for state in ("key-saved", "llm-error"):
            with self.subTest(state=state), tempfile.TemporaryDirectory(prefix="parley_test_settingswin_") as tmp:
                env = {**os.environ, "PARLEY_HOME": str(Path(tmp) / "parley_test_ui_smoke")}
                r = subprocess.run([sys.executable, "-m", "parley.ui_preview", "--smoke", "--page", "llm", "--state", state],
                                   capture_output=True, text=True, timeout=120, env=env, cwd=ROOT)
            if r.returncode == 1 and "could not start" in r.stderr:
                self.skipTest("no display for Tk")
            self.assertEqual(r.returncode, 0, r.stderr[-500:])

    def test_ui_preview_keys_live_in_memory_never_in_the_windows_credential_store(self):
        with tempfile.TemporaryDirectory(prefix="parley_test_settingswin_") as tmp:
            home = Path(tmp) / "parley_test_ui_keys"
            home.mkdir()
            with mock.patch.dict(os.environ, {"PARLEY_HOME": str(home)}):
                host, _home, _created = ui_preview.build_host("key-saved", home=home, page="llm")
        self.assertIsInstance(host.keys.backend, ui_preview.FakeKeyBackend)
        self.assertTrue(host.keys.exists("anthropic"))
        self.assertFalse(host.keys.exists("openai"))

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
