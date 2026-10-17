"""Speech-to-text with faster-whisper (CTranslate2), GPU when available."""
import logging
import time
from pathlib import Path

import numpy as np

from . import gpu
from .config import SttConfig

log = logging.getLogger(__name__)


class Transcriber:
    def __init__(self, cfg: SttConfig, model_dir: Path):
        self.cfg = cfg
        self.model_dir = str(model_dir)
        gpu.add_cuda_dll_dirs()
        import ctranslate2
        from faster_whisper import WhisperModel

        start = time.perf_counter()
        self.model = None
        if cfg.device == "cuda" or (cfg.device == "auto" and ctranslate2.get_cuda_device_count() > 0):
            try:
                self.name = self._model_name(gpu=True)
                self.model = self._load(WhisperModel, self.name, "cuda", cfg.compute_type)
                self._warm_up()
                where = "GPU"
            except Exception as e:
                log.warning("Whisper on GPU failed (%s); using the CPU instead", e)
                self.model = None
        if self.model is None:
            self.name = self._model_name(gpu=False)
            self.model = self._load(WhisperModel, self.name, "cpu", "int8")
            self._warm_up()
            where = "CPU"
        log.info("Speech recognition ready: %s on %s (%.1fs)", self.name, where, time.perf_counter() - start)

    def _model_name(self, gpu: bool) -> str:
        if self.cfg.model != "auto":
            return self.cfg.model
        if gpu:
            return "large-v3-turbo"
        return "small.en" if self.cfg.language == "en" else "small"  # ~1s per question on a laptop CPU

    def _load(self, WhisperModel, name: str, device: str, compute_type: str):
        # Stay offline once the model is cached; only the first run downloads it
        from huggingface_hub.errors import LocalEntryNotFoundError

        try:
            return WhisperModel(name, device=device, compute_type=compute_type,
                                download_root=self.model_dir, local_files_only=True)
        except LocalEntryNotFoundError:
            pass
        log.info("Downloading speech model %s, first run only ...", name)
        return WhisperModel(name, device=device, compute_type=compute_type, download_root=self.model_dir)

    def _warm_up(self) -> None:
        # Runs the encoder once so CUDA kernels are loaded before the first real question
        segments, _ = self.model.transcribe(np.zeros(16000, dtype=np.float32), language="en", vad_filter=False)
        list(segments)

    def transcribe(self, audio: np.ndarray) -> str:
        peak = float(np.abs(audio).max()) if len(audio) else 0.0
        if 0 < peak < 0.5:
            audio = audio * (0.5 / peak)  # quiet mics (e.g. raw XLR inputs) lose words otherwise
        segments, _ = self.model.transcribe(
            audio,
            language=self.cfg.language or None,
            beam_size=self.cfg.beam_size,
            vad_filter=True,  # drops silence, which is where Whisper hallucinates
            condition_on_previous_text=False,
            without_timestamps=True,
        )
        return " ".join(s.text.strip() for s in segments).strip()
