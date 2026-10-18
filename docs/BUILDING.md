# Building Parley

There is no compile step: Parley is a Python package (3.11 to 3.13) run with [uv](https://docs.astral.sh/uv/). `uv build`
produces a wheel using hatchling.

## Standalone Windows copy

A standalone Windows copy (no Python needed) is built with PyInstaller, which is a build-only
dependency group (never installed for normal use):

```
uv run --group build python scripts/build.py
```

It writes `dist\install\` (about 635 MB; the NVIDIA CUDA libraries are not bundled, see below):
`Parley.exe` (a small windowed launcher), `current.json` and `app\<version>\` with two executables that share one `_internal\` folder:
`parley-tray.exe` (windowed, no console: the tray app the launcher starts) and `parley-app.exe` (console: `--ask`, `--list-devices`,
`--self-check`). Run `dist\install\Parley.exe`, or `parley-app.exe --self-check` to test the build, or
`uv run python scripts/tray_smoke.py --exe dist\install\Parley.exe` to check the tray app (window, no console, single instance, clean exit). `--feed-url URL` bakes in
an update server (default: none, so the built copy never checks for updates). `dist\` and `build\` are
not committed. Release builds must be made in a clean environment (no CUDA toolkit on `PATH`).

## Installer

The single-file installer is made from that tree with [Inno Setup 6](https://jrsoftware.org/isinfo.php) (`winget install JRSoftware.InnoSetup`, per-user):

```
uv run python scripts/make_installer.py
```

It writes `dist\installer\Parley-Setup-<version>.exe` (about 186 MB, LZMA2 solid compression; about a minute) and prints its size and
sha256. The version comes from `parley.__version__`. `packaging/parley.iss` has a commented-out `SignTool` hook; the installer is not signed yet,
so Windows SmartScreen will warn. Silent install for testing: `Parley-Setup-<ver>.exe /VERYSILENT /SUPPRESSMSGBOXES /NORESTART /SP- /DIR=<dir> /LOG=<file>`.

## CUDA in packaged copies

On an NVIDIA GPU the built copy downloads the CUDA 12 libraries (cuBLAS, cuDNN, cuFFT, CUDA
runtime; 6 pinned wheels, about 1.6 GB) once, in the background, on the first run: Parley says so, keeps working on the
CPU meanwhile. When the download is verified, the tray app restarts itself once, as soon as it is idle (not while you are
talking or it is answering), so the models come up on the GPU; it is not restarted again on later starts. They go to `runtime\cuda12\` in the Parley home, so app updates keep
them. `scripts/cuda_pack.py` writes `src/parley/cuda_pack.json` (URL, sha256 and size of each wheel, straight from `uv.lock`;
`build.py` refreshes it, and a test fails if the committed copy differs). Downloads are checked against that hash and
size, only from `files.pythonhosted.org`; only the listed DLLs are unpacked. If the download fails (offline, disk full,
bad hash) Parley just stays on the CPU and tries again next start. Set `[gpu] download_cuda = false`, or force
`device = "cpu"` for speech-to-text and text-to-speech, to opt out. Source checkouts use the pip packages instead.

## Command line

```sh
uv run parley --ask "what's the capital of Peru?"   # typed question through the whole pipeline, no mic
uv run parley --test-mic                            # microphone check
uv run parley --list-devices                        # list mics/speakers for the config
uv run parley -v                                    # debug logging with latency timings
uv run parley --tray                                # Windows: run from the notification area (log: <home>\logs\parley.log)
```

For an installed copy, command-line use goes through `parley-app.exe` (in `app\<version>\`), not `Parley.exe`, which has no console.

The llama.cpp runtime is checked against a built-in size and SHA-256 for the pinned release before it is unpacked;
a mismatch is discarded and Parley says so (the speech and language models from Hugging Face are not checksum-pinned).
