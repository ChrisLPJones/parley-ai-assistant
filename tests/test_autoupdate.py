"""The update flow wired into the app (stdlib unittest): injected Updater against the local fixture server, a temp
install root, a fake app and a fake prompt (no real dialog, no Parley home, no network).

Run: uv run python -m unittest discover -s tests -v
"""
import os
import queue
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

from parley import __main__ as cli
from parley import autoupdate, install, launcher, update
from parley.app import Parley
from parley.autoupdate import UpdateFlow
from parley.config import Config
from test_install import OK, Base, seed, tree
from test_update import make_release


class FakeApp:
    def __init__(self, busy_polls: int = 0, forever: bool = False) -> None:
        self.busy_polls, self.forever = busy_polls, forever
        self.said: list[str] = []

    def is_busy(self) -> bool:
        if self.forever:
            return True
        if self.busy_polls > 0:
            self.busy_polls -= 1
            return True
        return False

    def say(self, text: str) -> None:
        self.said.append(text)


class Prompt:
    def __init__(self, answer: bool, app: FakeApp | None = None) -> None:
        self.answer, self.app, self.calls = answer, app, []

    def __call__(self, title: str, text: str) -> bool:
        if self.app is not None:
            self.app.busy_seen = self.app.is_busy()  # type: ignore[attr-defined]
        self.calls.append((title, text))
        return self.answer


class FlowBase(Base):
    def setUp(self):
        super().setUp()
        seed(self.root, "0.1.0", tree())

    def flow(self, app, prompt, updater="default", **kw) -> UpdateFlow:
        u = self.updater() if updater == "default" else updater
        kw.setdefault("poll", 0.01)
        self.launched: list[tuple[Path, list[str]]] = []
        return UpdateFlow(app, self.root, u, prompt=prompt, launch=lambda r, a: self.launched.append((r, a)),
                          version="0.1.0", **kw)

    def patch_install(self, self_check_ok=True):
        real = self.real_install

        def fake(root, updater, manifest, files, **kw):
            return real(root, updater, manifest, files, self_check=lambda d: self_check_ok, harden=lambda d: None)
        return mock.patch.object(install, "install_version", side_effect=fake)

    real_install = staticmethod(install.install_version)

    def pointer(self):
        return launcher.read_pointer(self.root / launcher.POINTER)


class PromptFlowTests(FlowBase):
    def test_yes_installs_and_requests_restart(self):
        self.release("0.2.0", tree(OK + b"#2"))
        app, prompt = FakeApp(), Prompt(True)
        flow = self.flow(app, prompt)
        with self.patch_install():
            flow.run_check()
        self.assertEqual(len(prompt.calls), 1)
        self.assertIn("0.2.0", prompt.calls[0][1])
        self.assertIn("Fixes.", prompt.calls[0][1])
        self.assertIn("you have 0.1.0", prompt.calls[0][1])
        p = self.pointer()
        self.assertEqual((p.version, p.previous, p.state), ("0.2.0", "0.1.0", "pending"))
        self.assertTrue(flow.restart_requested.is_set())
        self.assertTrue(any("0.2.0" in s for s in app.said))
        flow.restart()
        self.assertEqual(self.launched[0][0], self.root)

    def test_no_skips_and_changes_nothing(self):
        self.release("0.2.0", tree(OK + b"#2"))
        before = self.pointer()
        prompt = Prompt(False)
        flow = self.flow(FakeApp(), prompt)
        with self.patch_install() as m:
            flow.run_check()
        self.assertEqual(len(prompt.calls), 1)
        m.assert_not_called()
        self.assertFalse(flow.restart_requested.is_set())
        self.assertEqual(self.pointer(), before)
        self.assertFalse((self.root / "app" / "0.2.0").exists())

    def test_prompt_shown_once_per_launch(self):
        self.release("0.2.0", tree(OK + b"#2"))
        prompt = Prompt(False)
        flow = self.flow(FakeApp(), prompt)
        flow.start()
        flow.start()
        for t in threading.enumerate():
            if t.name == "update-check":
                t.join(10)
        self.assertEqual(len(prompt.calls), 1)

    def test_not_prompted_while_busy_then_prompted_when_idle(self):
        self.release("0.2.0", tree(OK + b"#2"))
        app = FakeApp(busy_polls=5)
        prompt = Prompt(False, app)
        self.flow(app, prompt).run_check()
        self.assertEqual(len(prompt.calls), 1)
        self.assertFalse(app.busy_seen)  # type: ignore[attr-defined]
        self.assertEqual(app.busy_polls, 0)

    def test_busy_forever_never_prompts_or_installs(self):
        self.release("0.2.0", tree(OK + b"#2"))
        app, prompt = FakeApp(forever=True), Prompt(True)
        flow = self.flow(app, prompt)
        t = threading.Thread(target=flow.run_check)
        t.start()
        t.join(0.2)
        self.assertTrue(t.is_alive())
        flow.stop()
        t.join(5)
        self.assertFalse(t.is_alive())
        self.assertEqual(prompt.calls, [])
        self.assertFalse(flow.restart_requested.is_set())

    def test_busy_again_after_yes_waits_before_installing(self):
        self.release("0.2.0", tree(OK + b"#2"))
        app = FakeApp()
        order: list[str] = []

        def prompt(title, text):
            app.busy_polls = 3  # the user started talking while the box was open
            order.append("prompt")
            return True

        real_is_busy = app.is_busy
        app.is_busy = lambda: (order.append("busy") if app.busy_polls else None, real_is_busy())[1]  # type: ignore
        flow = self.flow(app, prompt)
        with self.patch_install() as m:
            flow.run_check()
        self.assertEqual(order[:2], ["prompt", "busy"])
        m.assert_called_once()
        self.assertTrue(flow.restart_requested.is_set())

    def test_offline_is_silent(self):
        dead = update.Updater("http://127.0.0.1:1/parley/version.json", "0.1.0", state_path=self.state,
                              _allow_http_localhost=True)
        app, prompt = FakeApp(), Prompt(True)
        flow = self.flow(app, prompt, updater=dead)
        flow.run_check()
        self.assertEqual((prompt.calls, app.said), ([], []))
        self.assertFalse(flow.restart_requested.is_set())

    def test_up_to_date_and_older_feed_are_silent(self):
        make_release(self.srv, "0.1.0", tree())
        prompt = Prompt(True)
        self.flow(FakeApp(), prompt).run_check()
        self.assertEqual(prompt.calls, [])

    def test_rolled_back_version_is_not_offered_again(self):
        self.release("0.2.0", tree(OK + b"#2"))
        ptr = self.pointer()
        ptr.skip = ["0.2.0"]
        launcher.write_pointer(self.root, ptr)
        prompt = Prompt(True)
        self.flow(FakeApp(), prompt).run_check()
        self.assertEqual(prompt.calls, [])

    def test_failed_install_is_spoken_and_does_not_restart(self):
        self.release("0.2.0", tree(OK + b"#2"))
        app, prompt = FakeApp(), Prompt(True)
        flow = self.flow(app, prompt)
        before = self.pointer()
        with self.patch_install(self_check_ok=False):
            flow.run_check()
        self.assertFalse(flow.restart_requested.is_set())
        self.assertEqual(len(app.said), 1)
        self.assertIn("didn't install", app.said[0])
        self.assertEqual(self.pointer(), before)

    def test_no_updater_means_no_check(self):
        prompt = Prompt(True)
        flow = self.flow(FakeApp(), prompt, updater=None)
        flow.start()
        flow.run_check()
        self.assertEqual(prompt.calls, [])


class ConfirmAndNoticeTests(FlowBase):
    def pending_020(self):
        with self.patch_install():
            u, m, idx = self.release("0.2.0", tree(OK + b"#2"))
            install.install_version(self.root, u, m, idx)

    def test_confirm_after_ready_marks_new_version_good(self):
        self.pending_020()
        self.assertEqual(self.pointer().state, "pending")
        flow = UpdateFlow(FakeApp(), self.root, None, version="0.2.0")
        flow.start()
        self.assertEqual(self.pointer().state, "good")

    def test_confirm_does_not_touch_other_versions(self):
        self.pending_020()
        UpdateFlow(FakeApp(), self.root, None, version="0.1.0").start()
        self.assertEqual(self.pointer().state, "pending")

    def test_rollback_notice_spoken_once(self):
        launcher.write_pointer(self.root, launcher.Pointer("0.1.0", skip=["0.2.0"]))
        app = FakeApp()
        UpdateFlow(app, self.root, None, version="0.1.0").start()
        self.assertEqual(app.said, ["Update to 0.2.0 failed; staying on 0.1.0."])
        again = FakeApp()
        UpdateFlow(again, self.root, None, version="0.1.0").start()
        self.assertEqual(again.said, [])

    def test_no_notice_without_a_failed_newer_version(self):
        launcher.write_pointer(self.root, launcher.Pointer("0.2.0", skip=["0.1.5"]))
        app = FakeApp()
        UpdateFlow(app, self.root, None, version="0.2.0").start()
        self.assertEqual(app.said, [])


class OffModeTests(unittest.TestCase):
    def test_source_checkout_has_updates_off(self):
        self.assertIsNone(update.build_info())
        self.assertIsNone(autoupdate.install_root())
        self.assertIsNone(autoupdate.create(FakeApp(), Config()))

    def test_frozen_without_install_layout_is_off(self):
        with tempfile.TemporaryDirectory(prefix="parley_test_wire_") as d:
            exe = Path(d) / "a" / "b" / "parley-app.exe"
            with mock.patch.object(sys, "frozen", True, create=True), mock.patch.object(sys, "executable", str(exe)):
                self.assertIsNone(autoupdate.install_root())

    def test_check_disabled_in_config_still_confirms_but_never_checks(self):
        with tempfile.TemporaryDirectory(prefix="parley_test_wire_") as d:
            root = Path(d)
            seed(root, "0.1.0", tree())
            cfg = Config()
            cfg.updates.check = False
            with mock.patch.object(autoupdate, "install_root", return_value=root), \
                    mock.patch.object(update, "build_info", return_value=update.BuildInfo("https://x.example/parley/version.json")):
                flow = autoupdate.create(FakeApp(), cfg)
                self.assertIsNotNone(flow)
                self.assertIsNone(flow.updater)
                cfg.updates.check = True
                self.assertIsNotNone(autoupdate.create(FakeApp(), cfg).updater)

    def test_feed_url_is_not_a_config_setting(self):
        from dataclasses import fields
        self.assertEqual([f.name for f in fields(Config().updates)], ["check"])

    def test_prompt_text_trims_long_notes(self):
        m = mock.Mock(version="0.2.0", notes="x" * 5000)
        text = autoupdate.prompt_text(m, "0.1.0")
        self.assertLess(len(text), 1800)
        self.assertIn("Install it now?", text)

    def test_spawn_launcher_without_exe_is_quiet(self):
        with tempfile.TemporaryDirectory(prefix="parley_test_wire_") as d, mock.patch("subprocess.Popen") as popen:
            autoupdate.spawn_launcher(Path(d), [])
            popen.assert_not_called()


class SelfCheckTests(unittest.TestCase):
    def test_exit_zero_in_a_real_run_without_network_or_mic(self):
        with tempfile.TemporaryDirectory(prefix="parley_test_wire_") as home:
            env = {**os.environ, "PARLEY_HOME": home}
            r = subprocess.run([sys.executable, "-m", "parley", "--self-check"], env=env, capture_output=True,
                               text=True, timeout=120)
            self.assertEqual(r.returncode, 0, r.stderr[-500:])
            self.assertIn("self-check ok", r.stdout)

    def test_missing_library_gives_nonzero(self):
        import importlib
        real = importlib.import_module

        def fake(name, *a, **k):
            if name == "kokoro_onnx":
                raise ImportError("boom")
            return real(name, *a, **k)

        with mock.patch("importlib.import_module", side_effect=fake):
            self.assertEqual(cli.self_check(None), 1)

    def test_bad_config_gives_nonzero(self):
        with tempfile.TemporaryDirectory(prefix="parley_test_wire_") as d:
            bad = Path(d) / "config.toml"
            bad.write_text("this is = = not toml", "utf-8")
            self.assertEqual(cli.self_check(bad), 1)


class IsBusyTests(unittest.TestCase):
    def app(self, recording=False):
        app = Parley.__new__(Parley)
        app.recorder = mock.Mock(recording=recording)
        app._jobs, app._sentences = queue.Queue(), queue.Queue()
        app.player = mock.Mock(queue=queue.Queue())
        return app

    def test_idle(self):
        self.assertFalse(self.app().is_busy())

    def test_recording_is_busy(self):
        self.assertTrue(self.app(recording=True).is_busy())

    def test_each_stage_is_busy_until_task_done(self):
        for name in ("_jobs", "_sentences", "player"):
            app = self.app()
            q = app.player.queue if name == "player" else getattr(app, name)
            q.put(1)
            self.assertTrue(app.is_busy(), name)
            q.get()
            self.assertTrue(app.is_busy(), f"{name}: taken but unfinished")
            q.task_done()
            self.assertFalse(app.is_busy(), name)

    def test_no_mic_mode(self):
        app = self.app()
        app.recorder = None
        self.assertFalse(app.is_busy())

    def test_say_queues_a_sentence_for_the_current_turn(self):
        app = self.app()
        app.turn = 4
        app.say("hello")
        turn, text, _ = app._sentences.get_nowait()
        self.assertEqual((turn, text), (4, "hello"))


if __name__ == "__main__":
    unittest.main()
