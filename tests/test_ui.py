"""The UI thread host (ui.py): thread hop, ordering, errors, shutdown. The logic runs against a fake root (no display);
a few tests start real Tk and skip cleanly where it can't (a CI runner without a desktop).

Run: uv run python -m unittest discover -s tests -v
"""
import threading
import time
import unittest

from parley import ui


class FakeRoot:
    """Just enough of Tk for UiThread: after / mainloop / quit / withdraw / destroy, driven on the creating thread."""

    def __init__(self):
        self.thread = threading.current_thread()
        self.callbacks = []
        self.quit_flag = False
        self.destroyed = False

    def after(self, ms, fn):
        self.callbacks.append(fn)

    def mainloop(self):
        while not self.quit_flag:
            time.sleep(0.005)
            cbs, self.callbacks = self.callbacks, []
            for cb in cbs:
                cb()

    def quit(self):
        self.quit_flag = True

    def destroy(self):
        self.destroyed = True


class PostTests(unittest.TestCase):
    def setUp(self):
        self.roots = []

        def factory():
            r = FakeRoot()
            self.roots.append(r)
            return r

        self.ui = ui.UiThread(factory)
        self.addCleanup(self.ui.stop)

    def test_post_runs_on_the_ui_thread_not_the_caller(self):
        seen = []
        done = threading.Event()
        self.assertTrue(self.ui.post(lambda: (seen.append(threading.current_thread()), done.set())))
        self.assertTrue(done.wait(5))
        self.assertIsNot(seen[0], threading.current_thread())
        self.assertEqual(seen[0].name, "ui")
        self.assertIs(seen[0], self.roots[0].thread)  # the thread that owns the root

    def test_one_thread_for_all_posts_and_fifo_order(self):
        out, done = [], threading.Event()
        threads = set()
        for i in range(20):
            self.ui.post(lambda i=i: (out.append(i), threads.add(threading.current_thread())))
        self.ui.post(done.set)
        self.assertTrue(done.wait(5))
        self.assertEqual(out, list(range(20)))
        self.assertEqual(len(threads), 1)
        self.assertEqual(len(self.roots), 1)

    def test_posts_from_many_threads(self):
        count, lock, done = [0], threading.Lock(), threading.Event()

        def bump():
            with lock:
                count[0] += 1
                if count[0] == 50:
                    done.set()

        workers = [threading.Thread(target=lambda: [self.ui.post(bump) for _ in range(10)]) for _ in range(5)]
        [w.start() for w in workers]
        self.assertTrue(done.wait(5))

    def test_a_failing_task_is_logged_and_does_not_stop_the_queue(self):
        done = threading.Event()
        with self.assertLogs("parley.ui", "ERROR"):
            self.ui.post(lambda: 1 / 0)
            self.ui.post(done.set)
            self.assertTrue(done.wait(5))

    def test_call_returns_the_result_and_raises_the_exception(self):
        self.assertEqual(self.ui.call(lambda: threading.current_thread().name), "ui")
        with self.assertRaises(ZeroDivisionError):
            self.ui.call(lambda: 1 / 0)

    def test_stop_ends_the_thread_and_destroys_the_root_there(self):
        self.ui.start()
        thread = self.ui._thread
        self.ui.stop()
        self.assertFalse(thread.is_alive())
        self.assertTrue(self.roots[0].destroyed)

    def test_can_start_again_after_a_stop(self):
        self.ui.call(lambda: None)
        self.ui.stop()
        self.assertEqual(self.ui.call(lambda: 7), 7)
        self.assertEqual(len(self.roots), 2)

    def test_post_returns_false_when_tk_cannot_start(self):
        def broken():
            raise RuntimeError("no display")

        u = ui.UiThread(broken)
        with self.assertLogs("parley.ui", "WARNING"):
            self.assertFalse(u.post(lambda: None))
        self.assertIn("no display", u.error)
        with self.assertRaises(RuntimeError):
            u.call(lambda: None)


class HelperTests(unittest.TestCase):
    def tearDown(self):
        ui.theme_override = ui.scale_override = None

    def test_overrides_win(self):
        ui.theme_override, ui.scale_override = "dark", 150
        self.assertEqual(ui.wanted_theme(), "dark")
        self.assertEqual(ui.wanted_scale(), 1.5)
        ui.theme_override = "light"
        self.assertEqual(ui.wanted_theme(), "light")

    def test_system_values_are_sane(self):
        self.assertIn(ui.wanted_theme(), ("light", "dark"))
        self.assertGreaterEqual(ui.wanted_scale(), 1.0)

    def test_dpi_awareness_is_idempotent(self):
        first = ui.enable_dpi_awareness()
        self.assertEqual(ui.enable_dpi_awareness(), first)


def tk_available() -> bool:
    try:
        return ui.start()
    except Exception:
        return False


class RealTkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not tk_available():
            raise unittest.SkipTest(f"Tk cannot start here ({ui.last_error()})")

    @classmethod
    def tearDownClass(cls):
        ui.stop()

    def test_theme_scaling_and_fonts(self):
        def probe():
            ui.scale_override, ui.theme_override = 200, "dark"
            try:
                theme = ui.apply_theme(ui.root())
                import tkinter.font as tkfont

                return theme, float(ui.root().tk.call("tk", "scaling")), tkfont.nametofont("TkDefaultFont").metrics("linespace")
            finally:
                ui.scale_override = ui.theme_override = None

        theme, scaling, line200 = ui.call(probe)
        self.assertIn(theme, ("dark", "sun-valley-dark"))  # or the fallback theme's own name
        self.assertGreater(scaling, 1.0)  # exact values depend on the display (Tk rounds); only the ratio is checked below

        def probe100():
            ui.scale_override = 100
            try:
                ui.apply_theme(ui.root())
                import tkinter.font as tkfont

                return tkfont.nametofont("TkDefaultFont").metrics("linespace")
            finally:
                ui.scale_override = None

        def scaling100():
            ui.scale_override = 100
            try:
                ui.apply_theme(ui.root())
                return float(ui.root().tk.call("tk", "scaling"))
            finally:
                ui.scale_override = None

        self.assertAlmostEqual(scaling / ui.call(scaling100), 2.0, delta=0.1)  # 200 % is about twice 100 %
        self.assertGreater(line200, ui.call(probe100) * 1.4)  # the unit for layout grows with the scale

    def test_tk_objects_are_only_touched_on_the_ui_thread(self):
        name = ui.call(lambda: threading.current_thread().name)
        self.assertEqual(name, "ui")


if __name__ == "__main__":
    unittest.main()
