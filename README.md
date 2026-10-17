# Parley

A push-to-talk voice assistant that runs entirely on your own machine. Hold a key, talk, let go,
and a local LLM answers out loud. No cloud services, no accounts, nothing leaves your PC.

```
hold key ──► mic ──► Whisper (GPU) ──► local LLM (streaming) ──► Kokoro TTS ──► speakers
                                              │                        ▲
                                              └── sentence by sentence ┘
```

**Why it feels fast**

- Every model is loaded and warmed up at startup and stays resident in VRAM.
- The mic stream is always open with a short pre-roll, so recording starts instantly and your first word isn't clipped.
- The answer is spoken sentence by sentence while the LLM is still generating, so speech starts after the first sentence, not the whole reply.
- Pressing the hotkey again interrupts the current answer immediately.

## Components

Everything runs locally by default and is downloaded by Parley itself on first run into its own
folder. Nothing needs to be installed separately, and nothing already on the machine (Ollama, CUDA
toolkit, model caches) is used.

| Stage | What | Notes |
|---|---|---|
| Speech-to-text | [faster-whisper](https://github.com/SYSTRAN/faster-whisper) | `large-v3-turbo` on NVIDIA GPUs, `small.en` on CPU |
| LLM runtime | [llama.cpp](https://github.com/ggml-org/llama.cpp) `llama-server` (pinned build) | CUDA on NVIDIA, Vulkan on AMD/Intel, Metal on Mac |
| LLM model | small GGUF from Hugging Face | Qwen3.5 4B with a GPU, 2B otherwise |
| Text-to-speech | [Kokoro](https://github.com/thewh1teagle/kokoro-onnx) 82M | CUDA via ONNX Runtime, or CPU |

The local models are deliberately small so Parley runs on ordinary laptops. CUDA libraries come
from pip and llama.cpp's own release, so only a graphics driver is needed.

### Using Claude or OpenAI instead

For smarter answers, the LLM step can use a cloud API while speech recognition and the voice stay
local. Only the transcribed text of your question is sent.

```toml
# %LOCALAPPDATA%\parley\config.toml
[llm]
backend = "anthropic"   # or "openai"
```

Then set `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` in your environment (or `api_key` in the config).
Defaults are `claude-opus-5-5` (at low effort for quick replies) and OpenAI's `chat-latest`;
change either with `model`. Any other GGUF works locally via `model_repo` + `model_file`, and
`backend = "ollama"` / `"server"` uses a local server you already run.

## What you can ask it to do

Besides answering questions, Parley can act on your computer and look things up:

- **"What's in here?" / "How many files are in this folder?"** with File Explorer open (counts, sizes, file types)
- **"Read this"** with a file selected in Explorer, or "read notes.txt in Downloads"
- **"Find my CV"**, **"Open Downloads"**, **"Open Spotify"**, **"Open youtube.com"**
- **"What's the weather in Auckland?"** or anything current, via web search

"This" and "here" mean the folder and selection in the File Explorer window you're using.
Everything is read-only except opening things, and Parley only opens folders, documents, websites
and Start menu apps, never scripts or programs by path. Files that may hold secrets (SSH keys,
`.env`, credential stores) are off limits. With a cloud backend, whatever a tool reads is sent to
that provider as part of the conversation. Turn tools off with `[tools] enabled = false`.

## Requirements

- Windows 10/11 (Linux and macOS should work, untested)
- Runs on CPU-only laptops; an NVIDIA GPU makes speech recognition more accurate and everything faster.
- [uv](https://docs.astral.sh/uv/)
- Disk: about 2-5 GB depending on hardware

## Run

```sh
git clone https://github.com/ChrisLPJones/parley
cd parley
uv run parley
```

The first run downloads everything (progress is shown); later starts take a few seconds and are fully offline.
Hold **Pause** (default) and speak; release to send. Press it again while Parley is talking to interrupt.

```sh
uv run parley --ask "what's the capital of Peru?"   # typed question, no mic
uv run parley --list-devices                        # pick mic/speakers for the config
uv run parley -v                                    # debug logging with latency timings
```

## Files and configuration

Parley keeps everything in `%LOCALAPPDATA%\parley` (`~/.local/share/parley` on Linux,
`~/Library/Application Support/parley` on macOS), or `$PARLEY_HOME` if set:

```
config.toml        your settings (copy config.example.toml; every key is optional)
models/            whisper/, kokoro/, llm/
runtime/           llama.cpp build
logs/              llama-server.log
```

Delete the folder to uninstall completely.

## License

MIT
