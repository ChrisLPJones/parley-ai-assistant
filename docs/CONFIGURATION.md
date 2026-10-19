# Configuration

Most settings can be changed in the Settings window (see the [usage guide](USAGE.md)); the **Advanced** page has the ones
not on another page (pre-roll, speech language, beam size, devices, GPU layers, your own model file). Everything
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
| `[stt]` | `model`, `device`, `language`, `cpu_threads` | `model`: `auto` picks by hardware, or `tiny.en`, `base.en`, `small.en`, `small`, `medium`, `large-v3-turbo` (only these: Parley downloads each one itself and checks it against a pinned hash; other names such as `distil-large-v3` are no longer accepted); language `en` by default (`""` auto-detects); Whisper CPU threads (0 = auto) |
| `[llm]` | `backend` | `builtin` (default), `anthropic`, `openai`, `ollama`, `server` |
| `[llm]` | `preset`, `model_repo`, `model_file`, `model_path` | Which local model (builtin backend) |
| `[llm]` | `runtime` | `auto`, `cuda`, `vulkan` or `cpu` (builtin backend) |
| `[tools]` | `enabled`, `web_search` | Let the LLM use files, apps and web search (limited to read-only access, a fixed list of openable file types and public web addresses) |
| `[tts]` | `voice`, `speed`, `device`, `model` | Kokoro voice, speed, `auto`/`cuda`/`cpu`, and the Kokoro file: `default`, `fp16` or `int8` (smaller and quicker; their sound quality is not checked yet; needs a restart) |
| (restart) | | These need a restart of Parley, and Settings says so: `stt.model`, `stt.device`, `stt.compute_type`, `stt.cpu_threads`, `tts.device`, `tts.model`, `gpu.download_cuda`. All other keys apply when you save |
| `[gpu]` | `download_cuda` | Packaged copies only (default `true`): download the CUDA libraries (about 1.6 GB, once) on an NVIDIA GPU; the CPU is used until they are ready, then Parley restarts itself once, when idle, to use the GPU |
| `[updates]` | `check` | Installed copies only (default `true`): look for a newer version after startup and ask before installing; a source checkout never updates itself. The update server is fixed at build time, not configurable |

## Cloud backends

```toml
# %LOCALAPPDATA%\parley\config.toml
[llm]
backend = "anthropic"   # or "openai"
```

Then save your key on the **LLM** page of Settings (masked box, **Save key**; it shows only "Saved", never the key) or
with `parley --set-key anthropic` (or `openai`; the prompt hides what you type): it goes into the
Windows credential store, never into a file. Changing the backend, model, server address or tools in Settings reloads the
language model without a restart (when idle). `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` in the environment work too, a saved
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
| LLM model | small GGUF from Hugging Face | Qwen3.5 4B with a GPU, 2B otherwise; `preset = "medium"` (Qwen3.5 9B) for strong hardware |
| Text-to-speech | [Kokoro](https://github.com/thewh1teagle/kokoro-onnx) 82M | CUDA via ONNX Runtime on NVIDIA, otherwise CPU (already faster than real time) |

The local models are deliberately small so Parley runs on ordinary laptops, which means answers
are simple.

Speech and voice models are fetched the same verified way as the language models: by Parley's own list, from a fixed
source, checked against a known size and sha256 before use (first run downloads what is missing). Whisper models live in
`models/whisper/<name>/`. A Whisper cache made by an older Parley (`models--Systran--...`) is not used: Parley downloads the
checked copy once more. You can pick, download and delete speech, language and voice models in Settings > Models.

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
