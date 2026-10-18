import time
import unittest
from unittest import mock

import numpy as np

from parley import audio


class FakeStream:
    def __init__(self, **kwargs):
        self.writes: list[np.ndarray] = []

    def start(self): pass
    def stop(self): pass
    def close(self): pass

    def write(self, data):
        self.writes.append(np.array(data, copy=True))


def make_player(current=lambda t: True):
    with mock.patch.object(audio.sd, "OutputStream", FakeStream):
        return audio.Player(is_current=current)


def speechlike(n=24000):
    return (0.3 + 0.4 * np.sin(2 * np.pi * 200 * np.arange(n) / audio.SPEAKER_RATE)).astype(np.float32)


class PlayerTests(unittest.TestCase):
    player = None

    def tearDown(self):
        if self.player:
            self.player.close()

    def played(self, player):
        player.queue.join()
        return np.concatenate(player._stream.writes) if player._stream.writes else np.zeros(0)

    def test_clip_starts_with_a_fade_in_not_a_step(self):
        self.player = make_player()
        self.player.play(1, speechlike())
        out = self.player._stream.writes
        self.player.queue.join()
        first = next(w for w in out if np.any(w))  # skip keep-alive silence
        self.assertEqual(first[0], 0.0)
        self.assertLess(np.max(np.abs(np.diff(first[:audio.Player.RAMP * 2]))), 0.05)  # a raw 0.3 step would fail

    def test_cut_off_clip_ends_with_a_ramp_to_zero(self):
        state = {"cur": True}
        self.player = make_player(lambda t: state["cur"])
        stream = self.player._stream
        orig = stream.write
        def write(data):
            orig(data)
            if np.any(data):
                state["cur"] = False  # turn goes stale after the first audible block
        stream.write = write
        self.player.play(1, speechlike())
        self.player.queue.join()
        audible = [w for w in stream.writes if np.any(w)]
        self.assertEqual(len(audible), 2)  # one block plus the short ramp-down
        self.assertLess(abs(audible[-1][-1]), 0.02)

    def test_stale_clip_is_dropped(self):
        self.player = make_player(lambda t: False)
        self.player.play(1, speechlike())
        self.player.queue.join()
        self.assertFalse(any(np.any(w) for w in self.player._stream.writes))

    def test_idle_player_keeps_writing_silence(self):
        self.player = make_player()
        time.sleep(0.3)
        writes = self.player._stream.writes
        self.assertGreater(len(writes), 0)
        self.assertFalse(any(np.any(w) for w in writes))

    def test_tone_is_shaped_and_whole_length(self):
        t = audio.tone(660)
        self.assertEqual(len(t), audio.SPEAKER_RATE * 60 // 1000)
        self.assertEqual(t[0], 0.0)
        self.assertLess(abs(t[-1]), 0.01)

    def test_pop_beeps_are_soft_whole_and_differ_in_pitch(self):
        def crossings(x):
            return int(np.sum(np.signbit(x[:-1]) != np.signbit(x[1:])))

        start, stop = audio.start_beep(), audio.stop_beep()
        for b in (start, stop):
            self.assertEqual(len(b), audio.SPEAKER_RATE * 80 // 1000)
            self.assertLess(abs(b[0]), 0.001)
            self.assertLess(abs(b[-1]), 0.001)
            self.assertLessEqual(float(np.max(np.abs(b))), 0.25)
            self.assertTrue(np.all(np.isfinite(b)))
        self.assertGreater(crossings(start), crossings(stop))


if __name__ == "__main__":
    unittest.main()
