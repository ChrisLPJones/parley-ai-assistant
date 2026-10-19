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
        self.assertEqual(mv.status_text(m, self.st(), False, "saved"), "Chosen (restart Parley)")
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

    def test_speech_and_voice_are_read_only_with_a_note(self):
        for mid in ("stt-small.en", "tts-kokoro"):
            for state in ("installed", "not_installed"):
                a = self.acts(mid, state)
                self.assertEqual((a.download, a.cancel, a.use, a.delete), (False,) * 4, mid)
                self.assertEqual(a.note, mv.COMING_SOON)

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

    def test_speech_and_voice_rows_are_read_only(self):
        self.open_models(self.host_with())
        for mid in ("stt-small.en", "stt-large-v3-turbo", "tts-kokoro", "tts-kokoro-int8"):
            self.pick(mid)
            self.assertEqual(self.btns(), {"download": False, "cancel": False, "use": False, "delete": False}, mid)
            self.assertIn("next update", self.note())

    def test_use_sets_pending_values_and_save_writes_only_those_keys(self):
        self.open_models(self.host_with())
        self.pick("llm-small")
        self.assertTrue(self.btns()["use"])
        self.assertIn("restarting Parley", self.note())
        self.on_ui(self.win._model_use)
        self.assertEqual(self.row("llm-small")[2], "Chosen (press Save)")
        self.assertNotIn("llm", self.cfg_path.read_text("utf-8"))  # nothing is written before Save
        changes, ok = self.on_ui(self.win.collect)
        self.assertTrue(ok)
        self.assertEqual(changes, {"llm.preset": "small"})  # the empty repo / file / path already match config.toml
        self.on_ui(self.win.save)
        self.wait_banner("Saved. 1 setting needs a restart")
        out = self.cfg_path.read_text("utf-8")
        self.assertIn('preset = "small"', out)
        self.assertIn("# my favourite", out)
        self.assertEqual(self.row("llm-small")[2], "Chosen (restart Parley)")
        self.assertEqual(self.row("llm-tiny")[2], "In use")  # still what is running

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
