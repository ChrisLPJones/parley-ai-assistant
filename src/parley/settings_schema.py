"""One table entry per setting in config.py: type, label, help, limits, how a change takes effect.

The table drives validation (here), the settings window and the restart list. tests/test_settings_schema.py
fails when a Config field has no entry, so every setting stays mirrored. Validators raise ValueError with a
short sentence and return the value to store.
"""
import math
import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from .credstore import check_key_shape
from .llama_server import PRESETS, RUNTIMES, check_model_names

# How a saved change takes effect.
LIVE = "live"  # read from the config at use
HOTKEY = "hotkey"  # the key listener is re-pointed
SWAP_AUDIO = "swap_audio"  # a new Recorder / Player is built
REBUILD_CHAT = "rebuild_chat"  # a new Chat (history starts fresh)
RELOAD_LLM = "reload_llm"  # a new llama-server (builtin) and Chat
RESTART = "restart"  # needs a restart of Parley
NEXT_LAUNCH = "next_launch"  # only read at start, no prompt needed
APPLY_MODES = (LIVE, HOTKEY, SWAP_AUDIO, REBUILD_CHAT, RELOAD_LLM, RESTART, NEXT_LAUNCH)

BACKENDS = ("builtin", "anthropic", "openai", "ollama", "server")
DEVICES = ("auto", "cuda", "cpu")
COMPUTE_TYPES = ("default", "float16", "int8_float16", "int8")
EFFORTS = ("low", "medium", "high")
PRESET_CHOICES = ("auto", *PRESETS)
TTS_MODEL_CHOICES = ("default", "fp16", "int8")
STT_MODEL_NAMES = ("tiny.en", "base.en", "small.en", "small", "medium", "large-v3-turbo")  # = the catalog's Whisper keys
# Kokoro voice prefix (first letter) -> espeak language
VOICE_LANGS = {"a": "en-us", "b": "en-gb", "e": "es", "f": "fr-fr", "h": "hi", "i": "it", "j": "ja", "p": "pt-br", "z": "cmn"}


def voice_lang(voice: str) -> str | None:
    """tts.lang that goes with a Kokoro voice name such as af_heart, or None for an unknown prefix."""
    return VOICE_LANGS.get(voice[:1])


@dataclass(frozen=True)
class Setting:
    path: str  # "table.key", as in config.toml
    type: str  # bool | int | float | str | choice | hotkey | device | text | path | url
    label: str
    help: str
    apply: str
    choices: tuple = ()
    range: tuple | None = None  # (min, max) inclusive, for int / float
    zero_ok: bool = False  # int: 0 is also accepted although it is outside range (it means "off" / "server's own")
    advanced: bool = False

    def validate(self, value):
        return _VALIDATORS[self.type](self, value)


# -- validators ----------------------------------------------------------------------------------------------

def _bool(s, v):
    if not isinstance(v, bool):
        raise ValueError(f"{s.label}: must be on or off")
    return v


def _number(s, v, kind):
    if isinstance(v, bool) or not isinstance(v, (int, float) if kind is float else int):
        raise ValueError(f"{s.label}: must be a {'number' if kind is float else 'whole number'}")
    if kind is float:
        v = float(v)
        if not math.isfinite(v):
            raise ValueError(f"{s.label}: must be a number")
    lo, hi = s.range
    if not (lo <= v <= hi or (s.zero_ok and v == 0)):
        zero = " (or 0)" if s.zero_ok else ""
        raise ValueError(f"{s.label}: must be between {lo} and {hi}{zero}")
    return v


def _int(s, v):
    return _number(s, v, int)


def _float(s, v):
    return _number(s, v, float)


def _choice(s, v):
    if not isinstance(v, str) or v not in s.choices:
        raise ValueError(f"{s.label}: choose one of {', '.join(s.choices)}")
    return v


def _hotkey(s, v):
    if not isinstance(v, str) or not re.fullmatch(r"[A-Za-z0-9_]{1,30}", v.strip()):
        raise ValueError(f"{s.label}: not a key name")
    from pynput import keyboard

    from .__main__ import _parse_key

    try:
        key = _parse_key(v)
    except SystemExit as e:
        raise ValueError(f"{s.label}: {e}") from None
    if not isinstance(key, (keyboard.Key, keyboard.KeyCode)):  # e.g. "mro" is an attribute of the Key class
        raise ValueError(f"{s.label}: not a key name")
    return v.strip().lower()


def _device(s, v):
    """audio.*_device: None or "" = system default (stored as None); a name (substring) or a device index."""
    if v is None or v == "":
        return None
    if (isinstance(v, int) and not isinstance(v, bool) and 0 <= v < 10_000):
        return v
    if isinstance(v, str) and v.strip() and len(v) <= 200 and v.isprintable():
        return v
    raise ValueError(f"{s.label}: pick a device from the list")


def _text(s, v):
    """Free text up to range[1] characters (range[0] is the minimum, normally 0)."""
    lo, hi = s.range
    if not isinstance(v, str) or not lo <= len(v) <= hi:
        raise ValueError(f"{s.label}: must be at most {hi} characters")
    return v


def _pattern(regex: str, what: str, empty_ok: bool = True):
    def check(s, v):
        if not isinstance(v, str) or len(v) > 200:
            raise ValueError(f"{s.label}: {what}")
        if v == "" and empty_ok:
            return v
        if not re.fullmatch(regex, v, re.ASCII):
            raise ValueError(f"{s.label}: {what}")
        return v
    return check


def _model_repo(s, v):
    if not isinstance(v, str):
        raise ValueError(f"{s.label}: must be text")
    if v == "":
        return v
    try:
        check_model_names(v, "x.gguf")
    except ValueError as e:
        raise ValueError(f"{s.label}: {e}") from None
    return v


def _model_file(s, v):
    if not isinstance(v, str):
        raise ValueError(f"{s.label}: must be text")
    if v == "":
        return v
    try:
        check_model_names("x/x", v)
    except ValueError as e:
        raise ValueError(f"{s.label}: {e}") from None
    return v


def _model_path(s, v):
    if not isinstance(v, str) or len(v) > 1000 or "\0" in v:
        raise ValueError(f"{s.label}: not a file path")
    if v == "":
        return v
    p = Path(v)
    if p.suffix.lower() != ".gguf":
        raise ValueError(f"{s.label}: must be a .gguf file")
    if not p.is_file():
        raise ValueError(f"{s.label}: that file doesn't exist")
    return v


def _url(s, v):
    if not isinstance(v, str) or len(v) > 300:
        raise ValueError(f"{s.label}: not a web address")
    if v == "":
        return v
    try:
        u = urlparse(v)
        u.port  # raises ValueError for a bad port
    except ValueError:
        raise ValueError(f"{s.label}: not a web address") from None
    if u.scheme not in ("http", "https") or not u.hostname or re.search(r"\s", v):
        raise ValueError(f"{s.label}: must start with http:// or https://")
    if u.username or u.password:
        raise ValueError(f"{s.label}: must not contain a user name or password")
    return v


def _voice(s, v):
    # The list of installed voices needs Kokoro loaded; the window checks it, here only the shape.
    return _pattern(r"[a-z]{2}_[a-z0-9]{1,30}", "not a voice name like af_heart", empty_ok=False)(s, v)


_VALIDATORS = {"bool": _bool, "int": _int, "float": _float, "str": lambda s, v: _STR_CHECKS[s.path](s, v), "choice": _choice, "hotkey": _hotkey, "device": _device,
               "text": _text, "path": _model_path, "url": _url}


# -- the table -----------------------------------------------------------------------------------------------

def _s(path, type_, label, help_, apply, **kw):
    return Setting(path, type_, label, help_, apply, **kw)


SETTINGS: tuple[Setting, ...] = (
    _s("hotkey.key", "hotkey", "Talk key", "Hold this key to talk, release to send.", HOTKEY),
    _s("audio.input_device", "device", "Microphone", "Empty uses the system default.", SWAP_AUDIO),
    _s("audio.output_device", "device", "Speaker", "Empty uses the system default.", SWAP_AUDIO),
    _s("audio.preroll_ms", "int", "Pre-roll (ms)", "Audio kept from just before the key press so first words aren't clipped.",
       SWAP_AUDIO, range=(0, 2000), advanced=True),
    _s("audio.sounds", "bool", "Beeps", "Short beeps when recording starts and stops.", LIVE),
    _s("stt.model", "str", "Speech model", "auto, or a Whisper model from the Models page, such as small.en.", RESTART, advanced=True),
    _s("stt.device", "choice", "Speech device", "Where speech recognition runs.", RESTART, choices=DEVICES, advanced=True),
    _s("stt.compute_type", "choice", "Speech precision", "Compute type for speech recognition.", RESTART,
       choices=COMPUTE_TYPES, advanced=True),
    _s("stt.language", "str", "Spoken language", "Language code such as en; empty detects it.", LIVE),
    _s("stt.beam_size", "int", "Beam size", "Higher is slightly more accurate and slower.", LIVE, range=(1, 10), advanced=True),
    _s("stt.cpu_threads", "int", "CPU threads", "Threads for speech recognition on the CPU; 0 picks automatically.",
       RESTART, range=(0, 64), advanced=True),
    _s("llm.backend", "choice", "Language model", "Where answers come from.", RELOAD_LLM, choices=BACKENDS),
    _s("llm.preset", "choice", "Built-in model size", "Which built-in model; auto picks from your hardware.", RELOAD_LLM,
       choices=PRESET_CHOICES),
    _s("llm.model_repo", "str", "Hugging Face repo", "Use any GGUF instead of a preset, as owner/name.", RELOAD_LLM, advanced=True),
    _s("llm.model_file", "str", "Model file", "The .gguf file in that repo.", RELOAD_LLM, advanced=True),
    _s("llm.model_path", "path", "Model file on disk", "A .gguf file already on this computer.", RELOAD_LLM, advanced=True),
    _s("llm.runtime", "choice", "Built-in runtime", "Where the built-in model runs.", RELOAD_LLM, choices=RUNTIMES, advanced=True),
    _s("llm.gpu_layers", "int", "GPU layers", "-1 fits automatically to free memory.", RELOAD_LLM, range=(-1, 999), advanced=True),
    _s("llm.model", "str", "Model name", "Cloud or server model name; empty uses the default.", REBUILD_CHAT, advanced=True),
    _s("llm.api_key", "str", "API key", "Saved in the Windows credential store, not in config.toml (a key found there is moved at startup). "
       "Without one, ANTHROPIC_API_KEY / OPENAI_API_KEY is used.", REBUILD_CHAT, advanced=True),
    _s("llm.effort", "choice", "Claude effort", "Low answers fastest.", LIVE, choices=EFFORTS, advanced=True),
    _s("llm.base_url", "url", "Server address", "Override the endpoint for OpenAI, Ollama or another server.", REBUILD_CHAT,
       advanced=True),
    _s("llm.system_prompt", "text", "System prompt", "Instructions given to the model before every conversation.", LIVE,
       range=(0, 4000), advanced=True),
    _s("llm.temperature", "float", "Temperature", "Higher is more varied.", LIVE, range=(0, 2), advanced=True),
    _s("llm.num_ctx", "int", "Context size", "Context window in tokens (512 to 131072); for Ollama 0 keeps the server's.", RELOAD_LLM,
       range=(512, 131072), zero_ok=True, advanced=True),
    _s("llm.keep_alive", "str", "Keep loaded (Ollama)", "How long Ollama keeps the model loaded, such as 30m.", LIVE, advanced=True),
    _s("llm.think", "bool", "Let it think", "Reasoning models think first (much slower to start talking).", RELOAD_LLM, advanced=True),
    _s("llm.history_turns", "int", "Remembered exchanges", "Question and answer pairs kept in the conversation.", LIVE,
       range=(0, 100)),
    _s("llm.reset_after_idle_s", "int", "New conversation after (s)", "Start fresh after this long without a question.", LIVE,
       range=(0, 86400), advanced=True),
    _s("tools.enabled", "bool", "Computer tools", "Let the model look at files and folders and open things.", REBUILD_CHAT),
    _s("tools.web_search", "bool", "Web search", "Let the model search the web.", REBUILD_CHAT),
    _s("tts.voice", "str", "Voice", "The speaking voice.", LIVE),
    _s("tts.speed", "float", "Speaking speed", "1 is normal speed.", LIVE, range=(0.5, 2)),
    _s("tts.lang", "str", "Voice language", "Follows the voice; set automatically.", LIVE, advanced=True),
    _s("tts.device", "choice", "Voice device", "Where speech synthesis runs.", RESTART, choices=DEVICES, advanced=True),
    _s("tts.model", "choice", "Voice model", "Which Kokoro file to use; the smaller ones are quicker (their sound is not checked yet).",
       RESTART, choices=TTS_MODEL_CHOICES, advanced=True),
    _s("updates.check", "bool", "Check for updates", "Look for a newer version after startup (installed copies only).", NEXT_LAUNCH),
    _s("gpu.download_cuda", "bool", "Download GPU libraries", "On an NVIDIA GPU, download the CUDA libraries once (about 1.6 GB).",
       RESTART, advanced=True),
)

BY_PATH = {s.path: s for s in SETTINGS}
RESTART_PATHS = frozenset(s.path for s in SETTINGS if s.apply == RESTART)

# Each "str" setting has its own check.
_STR_CHECKS = {
    # Only `auto` and the catalog's Whisper models: every one is downloaded pinned and hash-checked (no free-form repo names)
    "stt.model": _pattern("auto|" + "|".join(re.escape(n) for n in STT_MODEL_NAMES), "choose auto or a model from the Models page",
                          empty_ok=False),
    "stt.language": _pattern(r"[a-z]{2,3}", "use a language code such as en, or leave empty"),
    "llm.model_repo": _model_repo,
    "llm.model_file": _model_file,
    "llm.model": _pattern(r"[\w.:/@-]+", "not a model name"),
    "llm.api_key": lambda s, v: _api_key(s, v),
    "llm.keep_alive": _pattern(r"-?\d+(ms|s|m|h)?", "use a duration such as 30m"),
    "tts.voice": _voice,
    "tts.lang": _pattern(r"[a-z]{2,3}(-[a-z]{2})?", "use a language code such as en-us", empty_ok=False),
}


def _api_key(s, v):
    # Printable ASCII, no spaces, at most 512 characters; "" clears it. The key itself is never echoed in an error.
    if v == "":
        return v
    if not check_key_shape(v):
        raise ValueError(f"{s.label}: not a valid key")
    return v


def validate(path: str, value):
    """Check one setting and return the value to store; ValueError (a short sentence) when it is not acceptable."""
    try:
        setting = BY_PATH[path]
    except KeyError:
        raise ValueError(f"unknown setting {path}") from None
    return setting.validate(value)
