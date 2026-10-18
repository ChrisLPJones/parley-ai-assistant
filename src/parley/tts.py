"""Text-to-speech with Kokoro (ONNX), plus helpers to turn streamed LLM text into speakable sentences."""
import logging
import re
import time
from pathlib import Path

import numpy as np

from . import gpu
from .config import TtsConfig
from .download import download

log = logging.getLogger(__name__)

MODEL_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/"
MODEL_FILES = ("kokoro-v1.0.onnx", "voices-v1.0.bin")


class Speaker:
    def __init__(self, cfg: TtsConfig, model_dir: Path):
        import onnxruntime as ort
        from kokoro_onnx import Kokoro

        self.cfg = cfg
        for name in MODEL_FILES:
            if not (model_dir / name).exists():
                log.info("Downloading voice model %s, first run only ...", name)
                download(MODEL_URL + name, model_dir / name)
        start = time.perf_counter()
        ort.set_default_logger_severity(3)  # errors only; ORT warns about harmless graph details
        if cfg.device not in ("auto", "cuda", "cpu"):
            raise SystemExit(f"Unknown tts.device {cfg.device!r}; choose auto, cuda or cpu")
        want_cuda = cfg.device == "cuda" or (cfg.device == "auto" and gpu.nvidia() is not None)
        if want_cuda and "CUDAExecutionProvider" in ort.get_available_providers():
            gpu.add_cuda_dll_dirs()
            # Every sentence has a different input length; an exhaustive cuDNN algorithm search
            # per new shape would add hundreds of ms, so pick algorithms heuristically
            gpu_providers = [("CUDAExecutionProvider", {"cudnn_conv_algo_search": "HEURISTIC"}), "CPUExecutionProvider"]
        else:
            gpu_providers = None
        model = str(model_dir / MODEL_FILES[0])
        voices = str(model_dir / MODEL_FILES[1])
        session = None
        if gpu_providers:
            try:
                session = ort.InferenceSession(model, providers=gpu_providers)
                self.kokoro = Kokoro.from_session(session, voices)
                self.synth("Ready.")  # kernels only fail at the first run, so warm up inside the try
            except Exception as e:
                log.warning("Voice on GPU failed (%s); using the CPU instead", e)
                session = None
        if session is None:
            session = ort.InferenceSession(model, providers=["CPUExecutionProvider"])
            self.kokoro = Kokoro.from_session(session, voices)
            self.synth("Ready.")
        log.info("Voice ready: %s on %s (%.1fs)", cfg.voice, session.get_providers()[0], time.perf_counter() - start)

    def synth(self, text: str, voice: str | None = None, speed: float | None = None, lang: str | None = None) -> np.ndarray:
        """Speak `text` in the configured voice; the optional arguments override it for one call (voice samples)."""
        samples, _rate = self.kokoro.create(text, voice=voice or self.cfg.voice, speed=speed or self.cfg.speed,
                                            lang=lang or self.cfg.lang)
        return samples.astype(np.float32)


def speakable(text: str) -> str:
    """Strip markdown the model may emit anyway, so it isn't read out."""
    text = re.sub(r"`+", "", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"(?m)^\s*(#+|[-*+>]|\d+\.)\s+", "", text)
    text = re.sub(r"[*_~#|]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


class SentenceSplitter:
    """Buffers streamed tokens and releases whole sentences. The first one may be short so
    speech starts as soon as possible; later ones are batched a little for smoother prosody."""

    BOUNDARY = re.compile(r"[.!?…]+[\"')\]]*\s+|[:;]\s*\n|\n\s*\n|\n(?=\s*(?:[-*+]|\d+\.)\s)")
    CLAUSE = re.compile(r"[,;:—]\s+")

    def __init__(self, first_min: int = 2, min_len: int = 40, max_len: int = 250):
        self.buf = ""
        self.first_min, self.min_len, self.max_len = first_min, min_len, max_len
        self.emitted = 0

    def feed(self, text: str) -> list[str]:
        self.buf += text
        out = []
        while True:
            need = self.first_min if not self.emitted else self.min_len
            cut = None
            for m in self.BOUNDARY.finditer(self.buf):
                if len(self.buf[:m.end()].strip()) >= need:
                    cut = m.end()
                    break
            if cut is None and not self.emitted:
                # Get the first words out quickly: a long opening sentence may break at a clause
                for m in self.CLAUSE.finditer(self.buf):
                    if m.end() >= 30:
                        cut = m.end()
                        break
            if cut is None and len(self.buf) > self.max_len:
                # Run-on text with no sentence end: break at the last comma or space
                cut = max(self.buf.rfind(", ", 0, self.max_len), self.buf.rfind(" ", 0, self.max_len)) + 1 or self.max_len
            if cut is None:
                return out
            self._emit(self.buf[:cut], out)
            self.buf = self.buf[cut:]

    def flush(self) -> list[str]:
        out: list[str] = []
        self._emit(self.buf, out)
        self.buf = ""
        return out

    def _emit(self, chunk: str, out: list[str]) -> None:
        chunk = speakable(chunk)
        if re.search(r"\w", chunk):
            out.append(chunk)
            self.emitted += 1
