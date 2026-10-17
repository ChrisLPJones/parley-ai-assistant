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
    # auto = large-v3-turbo on an NVIDIA GPU, small.en on CPU (fast enough for laptops);
    # or any faster-whisper model name: tiny / base / small / medium / large-v3-turbo / distil-large-v3
    model: str = "auto"
    device: str = "auto"  # auto | cuda | cpu
    compute_type: str = "default"  # default | float16 | int8_float16 | int8
    language: str = "en"  # "" = auto-detect
    beam_size: int = 1
    # CPU only: threads for Whisper. 0 = auto (about half the logical cores, max 8); CTranslate2's own default is 4
    cpu_threads: int = 0


@dataclass
class LlmConfig:
    # builtin:   Parley downloads and runs a small local model itself (default, works offline)
    # anthropic: Claude via the Anthropic API (needs ANTHROPIC_API_KEY or api_key)
    # openai:    OpenAI API (needs OPENAI_API_KEY or api_key)
    # ollama / server: a local server you already run (Ollama / any OpenAI-compatible server)
    backend: str = "builtin"
    preset: str = "auto"  # builtin: auto (picked from your hardware) | small (Qwen3.5 4B) | tiny (Qwen3.5 2B)
    model_repo: str = ""  # builtin: any GGUF on Hugging Face instead of a preset, e.g. "unsloth/Qwen3.5-9B-GGUF"
    model_file: str = ""  #   ...and the file in that repo, e.g. "Qwen3.5-9B-Q4_K_M.gguf"
    model_path: str = ""  # builtin: or a GGUF file already on disk
    # builtin: where llama.cpp runs. auto = CUDA on NVIDIA, Vulkan on a discrete AMD/Intel GPU, otherwise the CPU
    # (integrated GPUs are not faster than a modern CPU); or force cuda | vulkan | cpu. Intel GPUs: unverified.
    runtime: str = "auto"
    gpu_layers: int = -1  # builtin: layers on the GPU; -1 = fit automatically to free memory
    model: str = ""  # anthropic: default claude-opus-5-5; openai: default chat-latest; ollama/server: "" = first available
    api_key: str = ""  # anthropic/openai; leave empty to use the ANTHROPIC_API_KEY / OPENAI_API_KEY env var
    effort: str = "low"  # anthropic: low | medium | high; low answers fastest
    base_url: str = ""  # openai/ollama/server: override the endpoint
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    temperature: float = 0.7
    num_ctx: int = 8192  # context window; for ollama 0 keeps whatever the server uses
    keep_alive: str = "30m"  # ollama only: how long the model stays loaded between questions
    think: bool = False  # let reasoning models think before answering (much slower to start talking)
    history_turns: int = 10  # user/assistant exchanges remembered
    reset_after_idle_s: int = 300  # start a fresh conversation after this long with no question


@dataclass
class ToolsConfig:
    enabled: bool = True  # let the LLM look at files/folders and open things on this computer
    web_search: bool = True  # and search the web (DuckDuckGo locally; Claude uses Anthropic's search)


@dataclass
class TtsConfig:
    voice: str = "af_heart"
    speed: float = 1.1
    lang: str = "en-us"
    # auto = CUDA on NVIDIA, otherwise CPU (Kokoro's DirectML build fails on this model, so AMD/Intel GPUs aren't used)
    device: str = "auto"  # auto | cuda | cpu


@dataclass
class UpdatesConfig:
    # Installed (packaged) copies only: look for a newer version after startup and ask before installing it.
    # A source checkout never updates itself. The update server is fixed at build time and cannot be set here.
    check: bool = True


@dataclass
class Config:
    hotkey: HotkeyConfig = field(default_factory=HotkeyConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    stt: SttConfig = field(default_factory=SttConfig)
    llm: LlmConfig = field(default_factory=LlmConfig)
    tools: ToolsConfig = field(default_factory=ToolsConfig)
    tts: TtsConfig = field(default_factory=TtsConfig)
    updates: UpdatesConfig = field(default_factory=UpdatesConfig)


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
