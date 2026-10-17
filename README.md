# Parley

A push-to-talk voice assistant that runs on your own machine. Hold a key, talk, let go, and a
local LLM answers out loud. By default speech recognition, the LLM and the voice all run on your
PC, with no accounts. Only web search and page fetches go online (see
[Cloud backends and privacy](#cloud-backends-and-privacy)).

```
hold key ──► mic ──► Whisper (GPU) ──► local LLM (streaming) ──► Kokoro TTS ──► speakers
                                              │                        ▲
                                              └── sentence by sentence ┘
```

**Why it feels fast**

- Every model is loaded and warmed up at startup and stays resident (in VRAM when there is an NVIDIA GPU).
- The mic stream is always open with a short pre-roll, so recording starts instantly and your first word isn't clipped.
- The answer is spoken sentence by sentence while the LLM is still generating, so speech starts after the first sentence, not the whole reply.
- Pressing the hotkey again interrupts the current answer immediately and starts a new recording.

Status: version 0.1.0, early. Windows 10/11 is the only tested platform.

## Using it

1. Start Parley (see [Install and run](#install-and-run-windows)) and wait for the console to show "Ready. Hold [pause] and speak; release to send. Ctrl+C to quit."
2. Hold **Pause** (the default hotkey), speak, and release to send.
3. Press the hotkey again while Parley is talking to interrupt it.

### What you can ask it to do

Besides answering questions, Parley can act on your computer and look things up:

- **"What's in here?" / "How many files are in this folder?"** with File Explorer open (counts, sizes, file types)
- **"Read this"** with a file selected in Explorer, or "read notes.txt in Downloads"
- **"Find my CV"**, **"Open Downloads"**, **"Open Spotify"**, **"Open youtube.com"**
- **"What's the weather in Auckland?"** or anything current, via web search

"This" and "here" mean the folder and selection in the File Explorer window you're using.
Everything is read-only except opening things, and Parley only opens folders, documents, websites
and Start menu apps, never scripts or programs by path. Files that may hold secrets (SSH keys,
`.env`, credential stores) are off limits. Turn tools off with `[tools] enabled = false`.

### Cloud backends and privacy

By default speech recognition, the LLM and the voice run locally. Only web search (DuckDuckGo) and
page fetches go online; they are on by default and `web_search = false` under `[tools]` turns them
off. Models are downloaded on first run. For smarter answers, the LLM step can use the Anthropic or
OpenAI API while speech recognition and the voice stay local. With a cloud backend, **only the
transcribed text and tool output leave your machine** (never audio), including anything a tool reads for the
conversation, such as the contents of a file you asked about.

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

## Install and run (Windows)

A packaged installer (`Parley-Setup-<version>.exe`, per-user, no administrator rights, installs to
`%LOCALAPPDATA%\Programs\Parley` with a Start menu entry) can be built (see [Build](#build)) but is not published anywhere yet; once it
is, you download it from the project's release or update site. Running a newer installer upgrades in place; uninstalling
removes the program but never your Parley data (`%LOCALAPPDATA%\parley`: models, settings, GPU libraries). The Start menu entry opens
a console window for now. Until the installer is published you run Parley from source.

Prerequisites:

- Windows 10 or 11
- [uv](https://docs.astral.sh/uv/)
- A microphone
- Optional: an NVIDIA graphics driver, for GPU speed and the more accurate speech model. Parley also runs on CPU-only laptops. You do not need the CUDA toolkit.

```sh
git clone https://github.com/ChrisLPJones/parley
cd parley
uv run parley
```

The first run downloads everything Parley needs (speech, language and voice models and the
llama.cpp runtime). This is several GB and progress is shown. Later starts take a few seconds and
work offline (except cloud backends and web search). The llama.cpp runtime is checked against a built-in size and
SHA-256 for the pinned release before it is unpacked; a mismatch is discarded and Parley says so (the speech and language
models from Hugging Face are not checksum-pinned).

```sh
uv run parley --ask "what's the capital of Peru?"   # typed question through the whole pipeline, no mic
uv run parley --test-mic                            # microphone check
uv run parley --list-devices                        # list mics/speakers for the config
uv run parley -v                                    # debug logging with latency timings
```

## Build

There is no compile step: Parley is a Python package (3.11 to 3.13) run with uv. `uv build`
produces a wheel using hatchling.

A standalone Windows copy (no Python needed) is built with PyInstaller, which is a build-only
dependency group (never installed for normal use):

```
uv run --group build python scripts/build.py
```

It writes `dist\install\` (about 620 MB; the NVIDIA CUDA libraries are not bundled, see below):
`Parley.exe` (a small launcher), `current.json` and `app\<version>\parley-app.exe` with its libraries.
Run `dist\install\Parley.exe`, or `parley-app.exe --self-check` to test the build. `--feed-url URL` bakes in
an update server (default: none, so the built copy never checks for updates). `dist\` and `build\` are
not committed. Release builds must be made in a clean environment (no CUDA toolkit on `PATH`). 

The single-file installer is made from that tree with [Inno Setup 6](https://jrsoftware.org/isinfo.php) (`winget install JRSoftware.InnoSetup`, per-user):

```
uv run python scripts/make_installer.py
```

It writes `dist\installer\Parley-Setup-<version>.exe` (about 173 MB, LZMA2 solid compression; about a minute) and prints its size and
sha256. The version comes from `parley.__version__`. `packaging/parley.iss` has a commented-out `SignTool` hook; the installer is not signed yet,
so Windows SmartScreen will warn. Silent install for testing: `Parley-Setup-<ver>.exe /VERYSILENT /SUPPRESSMSGBOXES /NORESTART /SP- /DIR=<dir> /LOG=<file>`.

**CUDA in packaged copies.** On an NVIDIA GPU the built copy downloads the CUDA 12 libraries (cuBLAS, cuDNN, cuFFT, CUDA
runtime; 6 pinned wheels, about 1.6 GB) once, in the background, on the first run: Parley says so, keeps working on the
CPU meanwhile, and uses the GPU from the next start. They go to `runtime\cuda12\` in the Parley home, so app updates keep
them. `scripts/cuda_pack.py` writes `src/parley/cuda_pack.json` (URL, sha256 and size of each wheel, straight from `uv.lock`;
`build.py` refreshes it, and a test fails if the committed copy differs). Downloads are checked against that hash and
size, only from `files.pythonhosted.org`; only the listed DLLs are unpacked. If the download fails (offline, disk full,
bad hash) Parley just stays on the CPU and tries again next start. Set `[gpu] download_cuda = false`, or force
`device = "cpu"` for speech-to-text and text-to-speech, to opt out. Source checkouts use the pip packages instead.

## Linux (experimental, untested)

Linux support has not been tested. Nothing here is known to work. To try it from source:

```sh
# Ubuntu: sudo apt install libportaudio2      Fedora: sudo dnf install portaudio
git clone https://github.com/ChrisLPJones/parley
cd parley
uv run parley
```

Expect problems:

- The global hotkey needs an X11 session; under Wayland it will likely fail.
- `pywin32` is a Windows-only dependency, and the Windows-only features (Explorer "this folder" and selection, GPU detection) are unavailable.
- Known open issues: the pip-installed CUDA libraries may not be found, because Parley extends `PATH` but not `LD_LIBRARY_PATH`; the pynput win32 filter is unverified on Linux; the Linux llama.cpp assets were checked against the release listing and hashed (Ubuntu Vulkan x64) but never run.

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
are simple. CUDA libraries come from pip and llama.cpp's own release, so only a graphics driver is needed.

### Which hardware is used

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

## Configuration

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
logs/              llama-server.log
```

To remove everything, delete this folder and the cloned repo. Key settings (all in `config.example.toml`):

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

## Troubleshooting

- **No Pause key (most laptops):** set a different key under `[hotkey]`, for example `key = "f13"`, `"scroll_lock"` or `"ctrl_r"`.
- **Nothing happens when you speak:** run `uv run parley --test-mic`, and use `--list-devices` to set `input_device` under `[audio]`.
- **Garbled or nonsense answers on a GPU without a crash (for example Vulkan):** set `runtime = "cpu"` under `[llm]`.
- **Need more detail:** run with `-v` for debug logging and latency timings; the local LLM log is in `logs/llama-server.log` in the Parley home.

## Licence

MIT
