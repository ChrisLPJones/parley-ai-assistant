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
                started = []
                p._start_llm_reload = started.append  # never load a real language model here
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
                elif s.apply in (S.REBUILD_CHAT, S.RELOAD_LLM):  # a new Chat is built in the background from a candidate copy
                    self.assertEqual((applied, restart), ([], []))
                    self.assertEqual(len(started), 1)
                    self.assertEqual(started[0]["paths"], [s.path])
                    table = "tools" if s.path.startswith("tools.") else "llm"
                    self.assertEqual(getattr(started[0][table], s.path.split(".")[1]), new_value)
                    self.assertEqual(get(p.cfg, s.path), before)  # the running value changes only when the new one loaded
                    self.assertEqual(p.restart_pending(), [])
                else:  # restart
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


class FakeChat:
    def __init__(self, name, fail=None):
        self.name, self.fail, self.cfg = name, fail, None
        self.provider = SimpleNamespace(describe=lambda: name, cfg=None)
        self.history = [{"role": "user", "content": "hi"}] if name == "old" else []

    def warm_up(self):
        if self.fail:
            raise self.fail

    def rebind(self, cfg):
        self.cfg = self.provider.cfg = cfg


SECRET = "sk-ant-" "api03-TESTKEY0123456789abcdef"


class ReloadTests(unittest.TestCase):
    """reload_llm / rebuild_chat / key_changed with a fake create_chat and a fake llama-server (no model, no network)."""

    def setUp(self):
        self.p = make_parley()
        self.old = self.p.chat = FakeChat("old")
        self.built, self.fail, self.gate = [], None, None
        self.starts, self.stops = [], []

        def create(llm, tools, base_url=None, local_model="local"):
            if self.gate is not None:
                self.gate.wait(5)
            chat = FakeChat(f"new-{llm.backend}-{llm.model or '-'}", self.fail)
            self.built.append((llm, tools, base_url, local_model, chat))
            return chat

        def start(llm, home):
            self.starts.append(llm.think)
            return f"http://127.0.0.1:{9000 + len(self.starts)}/v1", "model"

        for patcher in (mock.patch("parley.app.create_chat", side_effect=create),
                        mock.patch("parley.llama_server.start", side_effect=start),
                        mock.patch("parley.llama_server.stop", side_effect=lambda url: self.stops.append(url))):
            patcher.start()
            self.addCleanup(patcher.stop)

    def wait_for(self, want, timeout=5):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if self.p.llm_state[0] in want:
                return self.p.llm_state
            time.sleep(0.02)
        self.fail(f"llm_state stayed {self.p.llm_state}")

    def test_idle_change_swaps_in_a_new_chat_and_mutates_the_shared_config(self):
        shared = self.p.cfg.llm
        self.p.apply_config(with_change("llm.model", "gpt-test"))
        self.assertEqual(self.wait_for(("done", "error"))[0], "done")
        self.assertIsNot(self.p.chat, self.old)
        self.assertEqual(self.p.chat.history, [])  # history starts fresh
        self.assertIs(self.p.cfg.llm, shared)
        self.assertEqual(shared.model, "gpt-test")
        self.assertIs(self.p.chat.cfg, shared)  # live settings keep working through the new Chat
        self.assertEqual(self.p.restart_pending(), [])
        self.assertEqual(self.p.deferred, [])

    def test_tools_change_rebuilds_with_the_new_tools_config(self):
        self.p.apply_config(with_change("tools.web_search", False))
        self.wait_for(("done", "error"))
        self.assertFalse(self.built[0][1].web_search)
        self.assertFalse(self.p.cfg.tools.web_search)

    def test_failure_keeps_the_old_chat_and_says_so_in_a_plain_sentence(self):
        self.fail = SystemExit(f"backend rejected {SECRET}")
        self.p.apply_config(with_change("llm.model", "gpt-test"))
        state, text = self.wait_for(("done", "error"))
        self.assertEqual(state, "error")
        self.assertIs(self.p.chat, self.old)
        self.assertEqual(self.p.cfg.llm.model, "")  # the running config is untouched
        self.assertEqual(self.p.restart_pending(), ["llm.model"])
        self.assertNotIn(SECRET, text)
        self.assertFalse(self.p._llm_loading)
        # the next save can try again
        self.fail = None
        self.p.apply_config(with_change("llm.model", "gpt-test2"))
        self.assertEqual(self.wait_for(("done",))[0], "done")
        self.assertEqual(self.p.restart_pending(), [])

    def test_failure_text_never_carries_an_unexpected_exception_message(self):
        self.fail = RuntimeError(f"upstream said {SECRET}")
        self.p.apply_config(with_change("llm.model", "x"))
        state, text = self.wait_for(("error",))
        self.assertNotIn(SECRET, text)
        self.assertIn("RuntimeError", text)

    def test_busy_defers_until_the_answer_ends(self):
        self.p = p = make_parley(busy=True)
        p.chat = self.old
        applied, restart = p.apply_config(with_change("llm.model", "x"))
        self.assertEqual((applied, restart), ([], []))
        self.assertEqual(p.deferred, ["llm.model"])
        time.sleep(0.5)
        self.assertEqual(self.built, [])  # nothing started during the answer
        self.assertIs(p.chat, self.old)
        p._jobs.unfinished_tasks = 0
        self.assertEqual(self.wait_for(("done",))[0], "done")
        self.assertEqual(p.deferred, [])

    def test_swap_waits_when_an_answer_starts_while_the_new_chat_loads(self):
        self.gate = threading.Event()
        self.p.apply_config(with_change("llm.model", "x"))
        self.assertEqual(self.p.llm_state[0], "loading")
        self.p._jobs.unfinished_tasks = 1  # the user asks something meanwhile
        self.gate.set()
        time.sleep(0.6)
        self.assertIs(self.p.chat, self.old)  # the old Chat finishes its answer first
        self.assertEqual(self.p.llm_state[0], "loading")
        self.p._jobs.unfinished_tasks = 0
        self.assertEqual(self.wait_for(("done",))[0], "done")

    def test_a_second_save_waits_for_the_load_in_progress(self):
        self.gate = threading.Event()
        self.p.apply_config(with_change("llm.model", "first"))
        self.p.apply_config(with_change("llm.model", "second"))
        self.assertEqual(self.p.deferred, ["llm.model"])
        self.assertEqual(len(self.built), 0)
        self.gate.set()
        end = time.monotonic() + 5
        while len(self.built) < 2 and time.monotonic() < end:
            time.sleep(0.02)
        self.assertEqual([b[0].model for b in self.built], ["first", "second"])
        self.wait_for(("done",))

    def test_builtin_starts_a_new_server_only_when_its_settings_changed(self):
        p = self.p
        p._server = ("http://127.0.0.1:9000/v1", "model")
        p.apply_config(with_change("llm.think", True))  # a llama-server flag
        self.wait_for(("done",))
        self.assertEqual(self.starts, [True])
        self.assertEqual(p._server[0], "http://127.0.0.1:9001/v1")
        self.assertEqual(self.stops, ["http://127.0.0.1:9000/v1"])  # the old one stops only after the swap
        self.assertEqual(self.built[-1][2], "http://127.0.0.1:9001/v1")
        p.llm_state = ("idle", "")
        new = copy.deepcopy(p.cfg)
        new.tools.web_search = False  # a Chat-only change reuses the running server
        p.apply_config(new)
        self.wait_for(("done",))
        self.assertEqual(self.starts, [True])
        self.assertEqual(self.built[-1][2], "http://127.0.0.1:9001/v1")
        self.assertEqual(self.stops, ["http://127.0.0.1:9000/v1"])

    def test_failure_stops_a_server_it_started_and_keeps_the_old_one(self):
        p = self.p
        p._server = ("http://127.0.0.1:9000/v1", "model")
        self.fail = RuntimeError("llama-server exited during startup")
        p.apply_config(with_change("llm.think", True))
        self.wait_for(("error",))
        self.assertEqual(self.stops, ["http://127.0.0.1:9001/v1"])
        self.assertEqual(p._server, ("http://127.0.0.1:9000/v1", "model"))
        self.assertIs(p.chat, self.old)

    def test_switching_to_a_cloud_backend_stops_the_builtin_server_after_the_swap(self):
        p = self.p
        p._server = ("http://127.0.0.1:9000/v1", "model")
        p.apply_config(with_change("llm.backend", "anthropic"))
        self.wait_for(("done",))
        self.assertEqual(self.starts, [])
        self.assertIsNone(p._server)
        self.assertEqual(self.stops, ["http://127.0.0.1:9000/v1"])

    def test_key_changed_rebuilds_only_when_that_backend_runs(self):
        p = self.p
        p.cfg.llm.backend = "anthropic"
        p.key_changed("openai")
        time.sleep(0.2)
        self.assertEqual(self.built, [])
        p.key_changed("anthropic")
        self.assertEqual(self.wait_for(("done",))[0], "done")
        self.assertEqual(len(self.built), 1)
        self.assertEqual(self.starts, [])
        self.assertIsNot(p.chat, self.old)
        self.assertFalse(p._key_dirty)

    def test_key_changed_while_busy_waits_for_the_answer(self):
        self.p = p = make_parley(busy=True)
        p.chat = self.old
        p.cfg.llm.backend = "anthropic"
        p.key_changed("anthropic")
        time.sleep(0.4)
        self.assertEqual(self.built, [])
        p._jobs.unfinished_tasks = 0
        self.assertEqual(self.wait_for(("done",))[0], "done")

    def test_the_swap_never_copies_an_api_key_into_the_running_config(self):
        new = with_change("llm.model", "x")
        new.llm.api_key = SECRET
        self.p.apply_config(new)
        self.wait_for(("done",))
        self.assertEqual(self.p.cfg.llm.api_key, "")


class DescribeFailureTests(unittest.TestCase):
    def test_sentences(self):
        import httpx
        from parley.app import describe_failure

        self.assertEqual(describe_failure(SystemExit("Anthropic rejected the API key.")), "Anthropic rejected the API key.")
        self.assertIn("connect", describe_failure(httpx.ConnectError(f"boom {SECRET}")))
        self.assertIn("built-in", describe_failure(RuntimeError("llama-server exited during startup:\n  tail")))
        self.assertEqual(describe_failure(KeyError(SECRET)), "The language model could not be loaded (KeyError).")
        self.assertNotIn("startup", describe_failure(RuntimeError("llama-server exited during startup:\n  tail")))
        self.assertIn("not set up", describe_failure(SystemExit(2)))

    def test_key_shapes_are_scrubbed_even_from_our_own_sentences(self):
        from parley.app import describe_failure

        for text in (f"The server rejected {SECRET} today", "bad key " + "sk-" + "proj-" + "x" * 20 + " here"):
            out = describe_failure(SystemExit(text))
            self.assertNotIn("sk-", out)
            self.assertIn("[key]", out)
        self.assertEqual(describe_failure(SystemExit("task-queue-name failed")), "task-queue-name failed")
        self.assertLessEqual(len(describe_failure(SystemExit("x" * 900))), 200)


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
