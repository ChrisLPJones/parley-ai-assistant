# Configuration

Most settings can be changed in the Settings window (see the [usage guide](USAGE.md)). Everything
is also stored in plain files you can edit.

Parley keeps everything in a home folder, or `$PARLEY_HOME` if set:

| OS | Location |
|---|---|
| Windows | `%LOCALAPPDATA%\parley` |
| Linux | `~/.local/share/parley` |
| macOS | `~/Library/Application Support/parley` |

```
config.toml        your settings (copy config.example.toml; every key is optional)
models/            whisper/, kokoro/, llm/
runtime/           llama.cpp build
logs/              parley.log (tray app), llama-server.log
```

To remove everything, delete this folder (and the cloned repo, if you ran from source). Key settings
(all in `config.example.toml`):

| Section | Key | Meaning |
|---|---|---|
| `[hotkey]` | `key` | Hold-to-talk key, default `pause` (also `f13`, `scroll_lock`, `ctrl_r`, `insert`, or a single character) |
| `[audio]` | `input_device`, `output_device` | Name substring or index from `--list-devices` |
| `[stt]` | `model`, `device`, `language`, `cpu_threads` | `auto` picks by hardware; language `en` by default (`""` auto-detects); Whisper CPU threads (0 = auto) |
| `[llm]` | `backend` | `builtin` (default), `anthropic`, `openai`, `ollama`, `server` |
| `[llm]` | `preset`, `model_repo`, `model_file`, `model_path` | Which local model (builtin backend) |
| `[llm]` | `runtime` | `auto`, `cuda`, `vulkan` or `cpu` (builtin backend) |
| `[tools]` | `enabled`, `web_search` | Let the LLM use files, apps and web search |
| `[tts]` | `voice`, `speed`, `device` | Kokoro voice, speed, and `auto`/`cuda`/`cpu` |
| `[gpu]` | `download_cuda` | Packaged copies only (default `true`): download the CUDA libraries (about 1.6 GB, once) on an NVIDIA GPU; the CPU is used until they are ready |
| `[updates]` | `check` | Installed copies only (default `true`): look for a newer version after startup and ask before installing; a source checkout never updates itself. The update server is fixed at build time, not configurable |

## Cloud backends

```toml
# %LOCALAPPDATA%\parley\config.toml
[llm]
backend = "anthropic"   # or "openai"
```

Then save your key with `parley --set-key anthropic` (or `openai`; the prompt hides what you type): it goes into the
Windows credential store, never into a file. `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` in the environment work too, a saved
key wins. `parley --key-status` shows what is saved and which source is used; `parley --delete-key anthropic` removes it.
A plaintext `api_key` in an older `config.toml` is moved into the store at startup and removed from the file (rotate the
key if that file was ever synced or shared).
Defaults are `claude-opus-5-5` (at low effort for quick replies) and OpenAI's `chat-latest`;
change either with `model`. Any other GGUF works locally via `model_repo` + `model_file`, and
`backend = "ollama"` / `"server"` uses a local server you already run.

## Components

Speech, LLM and voice run locally by default. Parley downloads the models itself on first run into
its own folder. Nothing needs to be installed separately, and nothing already on the machine (Ollama, CUDA
toolkit, model caches) is used.

| Stage | What | Notes |
|---|---|---|
| Speech-to-text | [faster-whisper](https://github.com/SYSTRAN/faster-whisper) | `large-v3-turbo` on NVIDIA GPUs, `small.en` on CPU when language is "en", otherwise `small` (AMD and Intel GPUs can't run it, so the CPU does) |
| LLM runtime | [llama.cpp](https://github.com/ggml-org/llama.cpp) `llama-server` (pinned build) | CUDA on NVIDIA, Vulkan on a discrete AMD/Intel GPU, CPU otherwise, Metal on Mac |
| LLM model | small GGUF from Hugging Face | Qwen3.5 4B with a GPU, 2B otherwise |
| Text-to-speech | [Kokoro](https://github.com/thewh1teagle/kokoro-onnx) 82M | CUDA via ONNX Runtime on NVIDIA, otherwise CPU (already faster than real time) |

The local models are deliberately small so Parley runs on ordinary laptops, which means answers
are simple.

## Which hardware is used

Parley picks at startup and logs what it found (`Hardware: ...`). It never fails because of a GPU:
if a GPU path doesn't start it falls back to the CPU.

| Hardware | Speech-to-text | LLM | Text-to-speech | Status |
|---|---|---|---|---|
| NVIDIA GPU | CUDA | CUDA | CUDA | tested (RTX 5090) |
| Discrete AMD GPU (4 GB or more) | CPU | Vulkan | CPU | design only, not tested on a discrete card |
| Discrete Intel GPU (Arc) | CPU | Vulkan | CPU | **unverified**: no Intel hardware was available |
| Integrated GPU only (AMD or Intel), or no GPU | CPU | CPU | CPU | AMD integrated tested; Intel **unverified** |
| Intel CPU | CPU paths as above | | | **unverified** (same code as AMD CPUs; thread count tuning for hybrid P/E-core chips is untested) |

Integrated GPUs are not used for the LLM: on the test machine's small Radeon the first word took
about 10 s through Vulkan against about 1.2 s on the CPU. Kokoro's DirectML build (the only way to run
it on AMD/Intel GPUs) fails on this model, so voice stays on the CPU there.
