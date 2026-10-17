# Parley

A push-to-talk voice assistant that runs entirely on your own machine. Hold a key, talk, let go,
and a local LLM answers out loud. No cloud services, no accounts, nothing leaves your PC.

```
hold key ──► mic ──► Whisper (GPU) ──► local LLM (streaming) ──► Kokoro TTS ──► speakers
                                              │                        ▲
                                              └── sentence by sentence ┘
```

**Why it feels fast**

- Every model is loaded and warmed up at startup, and the LLM is kept resident in VRAM.
- The mic stream is always open with a short pre-roll, so recording starts instantly and your first word isn't clipped.
- The answer is spoken sentence by sentence while the LLM is still generating, so speech starts after the first sentence, not the whole reply.
- Pressing the hotkey again interrupts the current answer immediately.

## Components

| Stage | Default | Notes |
|---|---|---|
| Speech-to-text | [faster-whisper](https://github.com/SYSTRAN/faster-whisper) `large-v3-turbo` | CUDA float16; falls back to CPU int8 |
| LLM | [Ollama](https://ollama.com) | Or any OpenAI-compatible server (llama.cpp, LM Studio, vLLM) |
| Text-to-speech | [Kokoro](https://github.com/thewh1teagle/kokoro-onnx) 82M | Model files download on first run (~350 MB) |

## Requirements

- Windows 10/11 (Linux should work, untested)
- An NVIDIA GPU is strongly recommended. CPU works but speech recognition is much slower.
- [Ollama](https://ollama.com) with at least one model pulled, e.g. `ollama pull qwen3:8b`
- [uv](https://docs.astral.sh/uv/)

## Run

```sh
git clone https://github.com/ChrisLPJones/parley
cd parley
uv run parley
```

Hold **Pause** (default) and speak; release to send.

Useful flags:

```sh
uv run parley --ask "what's the capital of Peru?"   # typed question, no mic
uv run parley --list-devices                        # pick mic/speakers for the config
uv run parley -v                                    # debug logging with latency timings
```

## Configuration

Copy `config.example.toml` to `%APPDATA%\parley\config.toml` (or pass `--config path`) and edit.
Every setting is optional; see the example file for all of them.

## Choosing a model

Latency is dominated by how fast the LLM produces its first sentence. Smaller models answer faster;
reasoning ("thinking") models are fine as long as `think = false` (the default). With 24–32 GB of
VRAM a ~27B model at Q4/Q5 leaves room for Whisper and Kokoro; with 8–12 GB use a 4–8B model.

## License

MIT
