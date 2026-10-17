"""Settings, loaded from a TOML file layered over the defaults below."""
import logging
import os
import sys
import tomllib
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path

log = logging.getLogger(__name__)

DEFAULT_SYSTEM_PROMPT = (
    "You are Parley, a voice assistant. Everything you write is spoken aloud, so reply "
    "conversationally in plain sentences: no markdown, lists, headings, code blocks, or emoji. "
    "Keep answers short, usually one to three sentences, unless the user asks for more detail."
)


def app_dir() -> Path:
    """Where Parley keeps its config, downloaded runtime and models (several GB). PARLEY_HOME overrides."""
    if os.environ.get("PARLEY_HOME"):
        return Path(os.environ["PARLEY_HOME"])
    if sys.platform == "win32":
        return Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")) / "parley"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "parley"
    return Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share")) / "parley"


@dataclass
class HotkeyConfig:
    key: str = "pause"  # hold to talk; e.g. "pause", "f13", "scroll_lock", "ctrl_r"


@dataclass
class AudioConfig:
    input_device: str | int | None = None  # None = system default; name substring or index
    output_device: str | int | None = None
    preroll_ms: int = 300  # audio kept from just before the key press so first words aren't clipped
    sounds: bool = True  # short beeps when recording starts/stops


@dataclass
class SttConfig:
    model: str = "large-v3-turbo"  # any faster-whisper model name or local path
    device: str = "auto"  # auto | cuda | cpu
    compute_type: str = "default"  # default | float16 | int8_float16 | int8
    language: str = "en"  # "" = auto-detect
    beam_size: int = 1


@dataclass
class LlmConfig:
    # builtin: Parley downloads and runs llama.cpp itself (default, nothing to install).
    # ollama / openai: use a server you already run (Ollama, LM Studio, vLLM, llama-server, ...)
    backend: str = "builtin"
    preset: str = "auto"  # builtin: auto (sized to your GPU) | xl | large | medium | small | tiny
    model_repo: str = ""  # builtin: any GGUF on Hugging Face instead of a preset, e.g. "unsloth/Qwen3.5-9B-GGUF"
    model_file: str = ""  #   ...and the file in that repo, e.g. "Qwen3.5-9B-Q4_K_M.gguf"
    model_path: str = ""  # builtin: or a GGUF file already on disk
    gpu_layers: int = 999  # builtin: layers offloaded to the GPU (999 = all)
    base_url: str = ""  # ollama/openai: "" = http://127.0.0.1:11434 for ollama
    model: str = ""  # ollama/openai: "" = first model the server reports
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    temperature: float = 0.7
    num_ctx: int = 8192  # context window; for ollama 0 keeps whatever the server uses
    keep_alive: str = "30m"  # ollama only: how long the model stays loaded between questions
    think: bool = False  # let reasoning models think before answering (much slower to start talking)
    history_turns: int = 10  # user/assistant exchanges remembered
    reset_after_idle_s: int = 300  # start a fresh conversation after this long with no question


@dataclass
class TtsConfig:
    voice: str = "af_heart"
    speed: float = 1.1
    lang: str = "en-us"
    device: str = "auto"  # auto | cpu


@dataclass
class Config:
    hotkey: HotkeyConfig = field(default_factory=HotkeyConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    stt: SttConfig = field(default_factory=SttConfig)
    llm: LlmConfig = field(default_factory=LlmConfig)
    tts: TtsConfig = field(default_factory=TtsConfig)


def _apply(obj, data: dict, where: str) -> None:
    known = {f.name: f for f in fields(obj)}
    for key, value in data.items():
        if key not in known:
            log.warning("Unknown setting %s%s (ignored)", where, key)
            continue
        current = getattr(obj, key)
        if is_dataclass(current):
            if isinstance(value, dict):
                _apply(current, value, f"{where}{key}.")
            else:
                log.warning("Setting %s%s should be a table (ignored)", where, key)
        else:
            setattr(obj, key, value)


def default_config_path() -> Path:
    return app_dir() / "config.toml"


def load(path: Path | None = None) -> Config:
    cfg = Config()
    path = path or default_config_path()
    if path.exists():
        with open(path, "rb") as f:
            _apply(cfg, tomllib.load(f), "")
        log.info("Loaded config from %s", path)
    else:
        log.info("No config file at %s, using defaults", path)
    return cfg
