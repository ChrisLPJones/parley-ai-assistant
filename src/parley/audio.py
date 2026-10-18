"""Microphone capture with pre-roll, and an interruptible speaker queue."""
import collections
import logging
import queue
import threading
from typing import Callable

import numpy as np
import sounddevice as sd

log = logging.getLogger(__name__)

MIC_RATE = 16000  # what Whisper expects
SPEAKER_RATE = 24000  # what Kokoro produces


def _device(spec, kind: str):
    if spec is None or spec == "" or isinstance(spec, int):  # "" = system default
        return spec if spec != "" else None
    for i, dev in enumerate(sd.query_devices()):
        if spec.lower() in dev["name"].lower() and dev[f"max_{kind}_channels"] > 0:
            return i
    log.warning("No %s device matching %r, using default", kind, spec)
    return None


class Recorder:
    """Keeps the mic stream open so recording starts instantly, holding a short pre-roll."""

    def __init__(self, device=None, preroll_ms: int = 300):
        self._lock = threading.Lock()
        self._preroll = collections.deque(maxlen=max(1, preroll_ms * MIC_RATE // 1000 // 512))
        self._chunks: list[np.ndarray] | None = None
        self._stream = sd.InputStream(
            samplerate=MIC_RATE, channels=1, dtype="float32", blocksize=512,
            device=_device(device, "input"), callback=self._callback,
        )
        self._stream.start()

    def _callback(self, data, frames, time_info, status):
        block = data[:, 0].copy()
        with self._lock:
            if self._chunks is not None:
                self._chunks.append(block)
            else:
                self._preroll.append(block)

    @property
    def recording(self) -> bool:
        return self._chunks is not None

    def close(self) -> None:
        """Stop and release the microphone (used when the device is swapped from Settings)."""
        try:
            self._stream.stop()
            self._stream.close()
        except Exception:
            log.debug("Closing the input stream failed", exc_info=True)

    def start(self) -> None:
        with self._lock:
            self._chunks = list(self._preroll)

    def stop(self) -> np.ndarray:
        with self._lock:
            chunks, self._chunks = self._chunks or [], None
            self._preroll.clear()
        return np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)


def tone(freq: float, ms: int = 60, volume: float = 0.15) -> np.ndarray:
    t = np.arange(int(SPEAKER_RATE * ms / 1000)) / SPEAKER_RATE
    fade = np.minimum(1, np.minimum(t, t[::-1]) * 400)  # ~2.5ms fade in/out, no clicks
    return (np.sin(2 * np.pi * freq * t) * fade * volume).astype(np.float32)


class Player:
    """Plays queued clips back to back. Clips tagged with a stale turn id are dropped,
    and the clip playing is cut off within ~50ms when its turn becomes stale."""

    BLOCK = SPEAKER_RATE // 20
    RAMP = SPEAKER_RATE // 200  # 5ms fade at a clip's start and where a clip is cut off

    def __init__(self, device=None, is_current: Callable[[int], bool] = lambda turn: True):
        self.queue: queue.Queue[tuple[int, np.ndarray, Callable | None]] = queue.Queue()  # unfinished_tasks = still playing
        self._is_current = is_current
        self._stream = sd.OutputStream(
            samplerate=SPEAKER_RATE, channels=1, dtype="float32", device=_device(device, "output"),
        )
        self._stream.start()
        threading.Thread(target=self._run, daemon=True, name="player").start()

    def play(self, turn: int, samples: np.ndarray, on_start: Callable | None = None) -> None:
        self.queue.put((turn, samples, on_start))

    def close(self) -> None:
        """Stop the playback thread and release the speaker (used when the device is swapped from Settings)."""
        self.queue.put((-1, None, None))  # sentinel: the worker leaves and closes the stream

    def _run(self) -> None:
        while True:
            try:
                turn, samples, on_start = self.queue.get(timeout=0.5 * self.BLOCK / SPEAKER_RATE)
            except queue.Empty:
                # keep the device fed: a stream that runs dry makes the next clip start with a pop
                self._stream.write(np.zeros(self.BLOCK // 2, dtype=np.float32))
                continue
            try:
                if samples is None:
                    try:
                        self._stream.stop()
                        self._stream.close()
                    except Exception:
                        log.debug("Closing the output stream failed", exc_info=True)
                    return
                if not self._is_current(turn):
                    continue
                if on_start:
                    on_start()
                samples = np.asarray(samples, dtype=np.float32).copy()
                n = min(self.RAMP, len(samples))
                samples[:n] *= np.linspace(0, 1, n, endpoint=False, dtype=np.float32)  # no step at the start
                for i in range(0, len(samples), self.BLOCK):
                    if not self._is_current(turn):
                        # cut off mid-clip: ramp the next few samples down instead of stopping dead
                        tail = samples[i:i + self.RAMP]
                        if len(tail) and i > 0:
                            self._stream.write(tail * np.linspace(1, 0, len(tail), endpoint=False, dtype=np.float32))
                        break
                    self._stream.write(samples[i:i + self.BLOCK])
            finally:
                self.queue.task_done()
