"""Parley.apply_config / preview_voice / device close(), with fakes (no models, no audio device, no display).

Run: uv run python -m unittest discover -s tests -v
"""
import copy
import threading
import time
import unittest
from types import SimpleNamespace
from unittest import mock

from parley import audio, hotkey, settings_schema as S
from parley.app import Parley
from parley.config import Config

EXPECTED_RESTART = {"stt.model", "stt.device", "stt.compute_type", "stt.cpu_threads", "tts.device", "tts.model", "gpu.download_cuda"}


class Q:
    def __init__(self, n=0):
        self.unfinished_tasks = n


class FakeDevice:
    def __init__(self, kind, device, fail=False):
        self.kind, self.device, self.closed = kind, device, False
        self.queue = Q()  # a Player has one

    def close(self):
        self.closed = True


class FakeListener:
    def __init__(self):
        self.keys = []

    def set_key(self, name):
        if name == "bogus":
            raise SystemExit("Unknown hotkey")
        self.keys.append(name)


def make_parley(busy=False):
    p = object.__new__(Parley)
    p.cfg = Config()
    p.turn = 0
    p._jobs, p._sentences = Q(1 if busy else 0), Q()
    p.recorder = SimpleNamespace(recording=False, close=lambda: None)
    p.player = SimpleNamespace(queue=Q(), closed=False)
    p._init_apply_state()
    p.hotkey = FakeListener()
    p.made = []
    p.fail_audio = False

    def make_recorder(device, preroll):
        if p.fail_audio:
            raise OSError("no such device")
        d = FakeDevice("rec", (device, preroll))
        d.recording = False
        p.made.append(d)
        return d

    def make_player(device):
        if p.fail_audio:
            raise OSError("no such device")
        d = FakeDevice("play", device)
        p.made.append(d)
        return d

    p._make_recorder, p._make_player = make_recorder, make_player
    return p


def other(s, cur):
    if s.type == "bool":
        return not cur
    if s.type == "choice":
        return next(c for c in s.choices if c != cur)
    if s.type in ("int", "float"):
        lo, hi = s.range
        return hi if cur != hi else lo
    if s.type == "hotkey":
        return "f13"
    return (cur if isinstance(cur, str) else "") + "x"  # apply_config does not validate; the window does


def get(cfg, path):
    t, k = path.split(".")
    return getattr(getattr(cfg, t), k)


def with_change(path, value):
    new = Config()
    t, k = path.split(".")
    setattr(getattr(new, t), k, value)
    return new


class ClassificationTests(unittest.TestCase):
    def test_restart_list_is_the_documented_one(self):
        self.assertEqual(set(S.RESTART_PATHS), EXPECTED_RESTART)

    def test_every_setting_is_classified_by_its_mode(self):
        for s in S.SETTINGS:
            if s.path == "llm.api_key":
                continue
            with self.subTest(setting=s.path, mode=s.apply):
                p = make_parley()
                before = get(p.cfg, s.path)
                new_value = other(s, before)
                applied, restart = p.apply_config(with_change(s.path, new_value))
                if s.apply in (S.LIVE, S.NEXT_LAUNCH):
                    self.assertEqual((applied, restart), ([s.path], []))
                    self.assertEqual(get(p.cfg, s.path), new_value)  # mutated in place: Speaker / Chat see it
                elif s.apply == S.HOTKEY:
                    self.assertEqual((applied, restart), ([s.path], []))
                    self.assertEqual(p.hotkey.keys, [new_value])
                    self.assertEqual(p.cfg.hotkey.key, new_value)
                elif s.apply == S.SWAP_AUDIO:
                    self.assertEqual((applied, restart), ([s.path], []))
                    self.assertEqual(get(p.cfg, s.path), new_value)
                    self.assertTrue(p.made)
                else:  # restart (and, until their machinery lands, rebuild_chat / reload_llm)
                    self.assertEqual((applied, restart), ([], [s.path]))
                    self.assertEqual(get(p.cfg, s.path), before)  # the running value is not faked
                    self.assertEqual(p.restart_pending(), [s.path])

    def test_api_key_is_never_applied(self):
        p = make_parley()
        new = with_change("llm.api_key", "TESTKEY-not-a-real-secret-0123456789")
        self.assertEqual(p.apply_config(new), ([], []))
        self.assertEqual(p.cfg.llm.api_key, "")

    def test_no_change_no_work(self):
        p = make_parley()
        self.assertEqual(p.apply_config(copy.deepcopy(p.cfg)), ([], []))
        self.assertEqual(p.made, [])

    def test_shared_sub_dataclass_is_mutated_in_place(self):
        p = make_parley()
        tts = p.cfg.tts
        p.apply_config(with_change("tts.voice", "bf_emma"))
        self.assertIs(p.cfg.tts, tts)
        self.assertEqual(tts.voice, "bf_emma")

    def test_restart_paths_clear_when_edited_back(self):
        p = make_parley()
        p.apply_config(with_change("stt.model", "small.en"))
        self.assertEqual(p.restart_pending(), ["stt.model"])
        p.apply_config(Config())
        self.assertEqual(p.restart_pending(), [])


class IdleTests(unittest.TestCase):
    def test_audio_only_swaps_what_changed(self):
        p = make_parley()
        old_player = p.player
        p.apply_config(with_change("audio.input_device", "USB mic"))
        self.assertEqual([d.kind for d in p.made], ["rec"])
        self.assertIs(p.player, old_player)
        self.assertEqual(p.recorder.device, ("USB mic", 300))

    def test_old_devices_are_closed_after_the_new_ones_exist(self):
        p = make_parley()
        old_rec_closed = []
        p.recorder = SimpleNamespace(recording=False, close=lambda: old_rec_closed.append(True))
        p.apply_config(with_change("audio.input_device", "USB mic"))
        self.assertEqual(old_rec_closed, [True])

    def test_failed_swap_keeps_the_old_device_and_asks_for_a_restart(self):
        p = make_parley()
        old = p.player
        p.fail_audio = True
        applied, restart = p.apply_config(with_change("audio.output_device", "Gone"))
        self.assertEqual((applied, restart), ([], ["audio.output_device"]))
        self.assertIs(p.player, old)
        self.assertIsNone(p.cfg.audio.output_device)
        self.assertEqual(p.restart_pending(), ["audio.output_device"])

    def test_unknown_hotkey_is_reported_as_restart(self):
        p = make_parley()
        applied, restart = p.apply_config(with_change("hotkey.key", "bogus"))
        self.assertEqual((applied, restart), ([], ["hotkey.key"]))
        self.assertEqual(p.cfg.hotkey.key, "pause")

    def test_no_listener_means_restart(self):
        p = make_parley()
        p.hotkey = None
        self.assertEqual(p.apply_config(with_change("hotkey.key", "f13")), ([], ["hotkey.key"]))

    def test_busy_defers_hotkey_and_audio_until_the_answer_ends(self):
        p = make_parley(busy=True)
        new = Config()
        new.hotkey.key, new.audio.output_device = "f13", "Speakers"
        applied, restart = p.apply_config(new)
        self.assertEqual((applied, restart), ([], []))
        self.assertEqual(p.deferred, ["hotkey.key", "audio.output_device"])
        self.assertEqual(p.hotkey.keys, [])
        p._jobs.unfinished_tasks = 0  # the answer finished
        end = time.monotonic() + 5
        while p.deferred and time.monotonic() < end:
            time.sleep(0.05)
        self.assertEqual(p.deferred, [])
        self.assertEqual(p.hotkey.keys, ["f13"])
        self.assertEqual(p.cfg.audio.output_device, "Speakers")

    def test_watcher_keeps_watching_when_busy_again_and_reports_applied(self):
        p = make_parley(busy=True)
        seen = []
        p.on_applied = seen.append
        real = p._apply_idle_changes
        flips = [True]

        def racy():  # busy again right after the watcher's own check, once
            if flips and not p.is_busy():
                flips.pop()
                p._jobs.unfinished_tasks = 1
            return real()

        p._apply_idle_changes = racy
        p.apply_config(with_change("hotkey.key", "f13"))
        p._jobs.unfinished_tasks = 0
        time.sleep(0.8)
        self.assertTrue(p.deferred)  # raced: still waiting, watcher alive
        p._jobs.unfinished_tasks = 0
        end = time.monotonic() + 5
        while p.deferred and time.monotonic() < end:
            time.sleep(0.05)
        self.assertEqual(p.hotkey.keys, ["f13"])
        time.sleep(0.2)
        self.assertEqual(seen, [p.cfg])  # the tray hears about the deferred change

    def test_a_later_save_replaces_a_waiting_change(self):
        p = make_parley(busy=True)
        p.apply_config(with_change("hotkey.key", "f13"))
        self.assertEqual(p.deferred, ["hotkey.key"])
        p.apply_config(Config())  # edited back to the running key
        self.assertEqual(p.deferred, [])
        p._jobs.unfinished_tasks = 0
        time.sleep(0.6)
        self.assertEqual(p.hotkey.keys, [])

    def test_live_settings_apply_even_while_busy(self):
        p = make_parley(busy=True)
        self.assertEqual(p.apply_config(with_change("tts.speed", 1.5)), (["tts.speed"], []))


class PreviewTests(unittest.TestCase):
    def make(self):
        p = make_parley()
        p.played = []
        p.gate = threading.Event()
        p.gate.set()

        def synth(text, voice=None, speed=None, lang=None):
            p.gate.wait(5)
            p.synthed = (text, voice, speed, lang)
            return [0.0]

        p.speaker = SimpleNamespace(kokoro=SimpleNamespace(get_voices=lambda: ["af_heart", "bf_emma"]), synth=synth)
        p.player = SimpleNamespace(queue=Q(), play=lambda turn, audio: p.played.append(turn))
        return p

    def wait_idle(self, p):
        end = time.monotonic() + 5
        while p._previewing and time.monotonic() < end:
            time.sleep(0.02)

    def test_plays_the_sample_with_the_voice_language(self):
        p = self.make()
        self.assertTrue(p.preview_voice("bf_emma", 1.3, "Hello"))
        self.wait_idle(p)
        self.assertEqual(p.synthed, ("Hello", "bf_emma", 1.3, "en-gb"))
        self.assertEqual(p.played, [0])

    def test_is_busy_while_synthesizing_and_refuses_a_second_one(self):
        p = self.make()
        p.gate.clear()
        self.assertTrue(p.preview_voice("af_heart", 1.0, "Hello"))
        self.assertTrue(p.is_busy())
        self.assertFalse(p.preview_voice("af_heart", 1.0, "Again"))
        p.gate.set()
        self.wait_idle(p)

    def test_refused_while_busy_with_an_answer(self):
        p = self.make()
        p._jobs = Q(1)
        self.assertFalse(p.preview_voice("af_heart", 1.0, "Hello"))

    def test_a_hotkey_press_cuts_it_off(self):
        p = self.make()
        p.gate.clear()
        p.preview_voice("af_heart", 1.0, "Hello")
        p.turn += 1  # begin_listening bumps the turn counter
        p.gate.set()
        self.wait_idle(p)
        self.assertEqual(p.played, [])

    def test_unknown_voice_and_bad_speed_are_refused(self):
        p = self.make()
        with self.assertRaises(ValueError):
            p.preview_voice("xx_nobody", 1.0, "Hello")
        with self.assertRaises(ValueError):
            p.preview_voice("af_heart", 9.0, "Hello")
        self.assertFalse(p.is_busy())


class DeviceCloseTests(unittest.TestCase):
    def test_player_close_stops_its_thread_and_the_stream(self):
        stream = mock.Mock()
        with mock.patch.object(audio.sd, "OutputStream", return_value=stream):
            player = audio.Player()
            threads = [t for t in threading.enumerate() if t.name == "player"]
            player.close()
            end = time.monotonic() + 3
            while stream.close.call_count == 0 and time.monotonic() < end:
                time.sleep(0.02)
        stream.stop.assert_called()
        stream.close.assert_called_once()
        for t in threads:
            t.join(3)
            self.assertFalse(t.is_alive())

    def test_recorder_close_releases_the_stream(self):
        stream = mock.Mock()
        with mock.patch.object(audio.sd, "InputStream", return_value=stream):
            audio.Recorder().close()
        stream.stop.assert_called()
        stream.close.assert_called_once()


class HotkeyNameTests(unittest.TestCase):
    def test_virtual_key_to_config_name(self):
        for vk, name in ((0x13, "pause"), (0xA3, "ctrl_r"), (0x41, "a"), (0x31, "1"), (0x7C, "f13"), (0x91, "scroll_lock")):
            self.assertEqual(hotkey.vk_to_name(vk), name)
        self.assertIsNone(hotkey.vk_to_name(0x6B))  # numpad plus: no pynput name

    def test_every_name_is_accepted_by_the_schema(self):
        for vk in (0x13, 0xA3, 0x41, 0x7C, 0x91, 0x20):
            self.assertEqual(S.validate("hotkey.key", hotkey.vk_to_name(vk)), hotkey.vk_to_name(vk))


if __name__ == "__main__":
    unittest.main()
