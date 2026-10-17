"""Speech-to-text with faster-whisper (CTranslate2), GPU when available."""
import importlib.util
import logging
import os
import sys
import time

import numpy as np

from .config import SttConfig

log = logging.getLogger(__name__)


def _add_cuda_dll_dirs() -> None:
    """Make the pip-installed cuBLAS/cuDNN DLLs findable so no CUDA toolkit install is needed."""
    if sys.platform != "win32":
        return
    for pkg in ("nvidia.cublas", "nvidia.cudnn"):
        try:
            spec = importlib.util.find_spec(pkg)
        except ModuleNotFoundError:
            continue
        for loc in (spec.submodule_search_locations or []) if spec else []:
            bin_dir = os.path.join(loc, "bin")
            if os.path.isdir(bin_dir):
                os.add_dll_directory(bin_dir)
                os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")


class Transcriber:
    def __init__(self, cfg: SttConfig):
        self.cfg = cfg
        _add_cuda_dll_dirs()
        from faster_whisper import WhisperModel

        start = time.perf_counter()
        try:
            self.model = self._load(WhisperModel, cfg.device, cfg.compute_type)
            self._warm_up()
        except Exception as e:
            if cfg.device == "cpu":
                raise
            log.warning("Whisper on GPU failed (%s); falling back to CPU, which is much slower", e)
            self.model = self._load(WhisperModel, "cpu", "int8")
            self._warm_up()
        log.info("Speech recognition ready: %s (%.1fs)", cfg.model, time.perf_counter() - start)

    def _load(self, WhisperModel, device: str, compute_type: str):
        # Stay offline once the model is cached; only the first run downloads it
        from huggingface_hub.errors import LocalEntryNotFoundError

        try:
            return WhisperModel(self.cfg.model, device=device, compute_type=compute_type, local_files_only=True)
        except LocalEntryNotFoundError:
            pass
        log.info("Downloading speech model %s (first run only) ...", self.cfg.model)
        return WhisperModel(self.cfg.model, device=device, compute_type=compute_type)

    def _warm_up(self) -> None:
        # Runs the encoder once so CUDA kernels are loaded before the first real question
        segments, _ = self.model.transcribe(np.zeros(16000, dtype=np.float32), language="en", vad_filter=False)
        list(segments)

    def transcribe(self, audio: np.ndarray) -> str:
        segments, _ = self.model.transcribe(
            audio,
            language=self.cfg.language or None,
            beam_size=self.cfg.beam_size,
            vad_filter=True,  # drops silence, which is where Whisper hallucinates
            condition_on_previous_text=False,
            without_timestamps=True,
        )
        return " ".join(s.text.strip() for s in segments).strip()
