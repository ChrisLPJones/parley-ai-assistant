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

Everything is downloaded by Parley itself on first run into its own folder. Nothing needs to be
installed separately, and nothing already on the machine (Ollama, CUDA toolkit, model caches) is used.

| Stage | What | Notes |
|---|---|---|
| Speech-to-text | [faster-whisper](https://github.com/SYSTRAN/faster-whisper) `large-v3-turbo` | CUDA float16; falls back to CPU int8 |
| LLM runtime | [llama.cpp](https://github.com/ggml-org/llama.cpp) `llama-server` (pinned build) | CUDA on NVIDIA, Vulkan on AMD/Intel, Metal on Mac |
| LLM model | GGUF from Hugging Face, sized to your GPU | see presets below |
| Text-to-speech | [Kokoro](https://github.com/thewh1teagle/kokoro-onnx) 82M | CUDA via ONNX Runtime, or CPU |

CUDA libraries come from pip (`nvidia-*-cu12` wheels) and llama.cpp's own release, so only an
NVIDIA driver is needed, not the CUDA toolkit.

### LLM presets

`preset = "auto"` picks the largest one that fits your VRAM alongside the speech models:

| Preset | Model | Download | Fits |
|---|---|---|---|
| xl | Qwen3.6 35B-A3B (MoE, 3B active) | 22 GB | 28 GB+ VRAM |
| large | Gemma 4 26B-A4B (MoE, 4B active) | 17 GB | 23 GB+ |
| medium | Qwen3.5 9B | 5.7 GB | 11 GB+ |
| small | Qwen3.5 4B | 2.7 GB | 8 GB+ |
| tiny | Qwen3.5 2B | 1.3 GB | CPU / small GPUs |

Any other GGUF works via `model_repo` + `model_file`, or `model_path` for a local file. If you
already run Ollama, LM Studio or similar, `backend = "ollama"` / `"openai"` uses that instead.

## Requirements

- Windows 10/11 (Linux and macOS should work, untested)
- An NVIDIA GPU is strongly recommended. Other GPUs run the LLM via Vulkan; speech runs on CPU.
- [uv](https://docs.astral.sh/uv/)
- Disk: about 3 GB for speech models and runtime, plus the LLM

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
