# PyInstaller spec for the executables. Run through scripts/build.py (it writes _build.py and parley.ico first), not directly.
#   parley-app : onedir app tree, goes to app/<version>/. Two exes share ONE _internal/ (one COLLECT):
#                  parley-app.exe  console=True : CLI, --ask, --self-check, launcher.CHILD_EXE, release smoke tests
#                  parley-tray.exe console=False: the notification-area app; its name implies --tray (tray_mode.implied())
#   Parley     : onefile launcher, stdlib only, windowed, sits in the install root; starts parley-tray.exe (launcher.RUN_EXE)
# No UPX (slower start, more antivirus hits). The nvidia-* CUDA wheels are NOT bundled (the CUDA pack is downloaded on first run: src/parley/cuda_pack.py).
import os
from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, collect_submodules

ROOT = os.path.dirname(SPECPATH)  # noqa: F821  (SPECPATH is provided by PyInstaller)
SRC = os.path.join(ROOT, "src")
ICON = os.path.join(ROOT, "build", "parley.ico")  # written by scripts/build.py (tray_icons.write_ico) before PyInstaller runs
if not os.path.isfile(ICON):
    raise SystemExit(f"{ICON} is missing: run scripts/build.py, not PyInstaller directly")

hidden = collect_submodules("parley") + [
    "pynput.keyboard._win32", "pynput.mouse._win32",  # chosen at run time by platform, not imported statically
    "win32com.shell.shell", "win32com.shell.shellcon", "pythoncom", "pywintypes",
    "win32gui", "win32cred", "win32api", "win32con", "win32event",  # tray window, credential store (also in --self-check)
    "tkinter", "tkinter.ttk", "tkinter.font", "sv_ttk",  # the settings window (parley.ui, parley.settings_window)
]
datas = []
binaries = []
for pkg in ("faster_whisper",       # assets/silero_vad*.onnx
            "kokoro_onnx",          # config.json
            "espeakng_loader",      # espeak-ng.dll and espeak-ng-data
            "phonemizer",
            "language_tags",
            "segments", "csvw", "babel", "tzdata",
            "certifi",              # httpx/anthropic CA bundle
            "ddgs", "primp",
            "sv_ttk"):              # sv.tcl + theme/ (the Windows 11 look of the settings window)
    try:
        datas += collect_data_files(pkg)
    except Exception:
        pass
binaries += collect_dynamic_libs("sounddevice") + collect_dynamic_libs("espeakng_loader")
datas += [(os.path.join(SRC, "parley", "cuda_pack.json"), "parley")]  # URL/sha256 of the CUDA pack wheels
datas += collect_data_files("_sounddevice_data")  # PortAudio DLL lives here
binaries += collect_dynamic_libs("_sounddevice_data")

EXCLUDES = ["nvidia", "setuptools", "pkg_resources", "matplotlib", "IPython", "pytest", "PyQt5", "PyQt6", "PySide6"]

# espeak-ng voice *variants* (voices/!v/*: "+whisper", "+croak" ...) are never requested by Kokoro, and "!" is not a
# character the update client's path rule accepts. The language voices (lang/, voices/ others) stay.
datas = [d for d in datas if "voices" + os.sep + "!v" not in d[1] and "voices/!v" not in d[1]]

app = Analysis(
    [os.path.join(ROOT, "packaging", "app_entry.py")],
    pathex=[SRC], binaries=binaries, datas=datas, hiddenimports=hidden, excludes=EXCLUDES,
)
# Drop the TensorRT provider (not used, large) from onnxruntime.
# Never bundle NVIDIA runtime DLLs that PyInstaller resolves from the venv's nvidia-* wheels (about 1 GB): the CUDA
# pack (downloaded on first run on NVIDIA machines) supplies them. onnxruntime_providers_cuda.dll itself stays.
CUDA_DLLS = ("cublas", "cufft", "cudart", "cudnn", "nvrtc", "nvjitlink", "curand", "cusparse", "cusolver", "nvfatbin")
app.binaries = [b for b in app.binaries
                if "providers_tensorrt" not in b[0].lower() and not os.path.basename(b[0]).lower().startswith(CUDA_DLLS)]
app_pyz = PYZ(app.pure)
app_exe = EXE(app_pyz, app.scripts, [], exclude_binaries=True, name="parley-app", console=True, upx=False, icon=ICON)
tray_exe = EXE(app_pyz, app.scripts, [], exclude_binaries=True, name="parley-tray", console=False, upx=False, icon=ICON)
COLLECT(app_exe, tray_exe, app.binaries, app.datas, name="parley-app", upx=False)

lau = Analysis(
    [os.path.join(ROOT, "packaging", "launcher_entry.py")],
    pathex=[SRC], excludes=EXCLUDES + ["numpy", "httpx", "faster_whisper", "onnxruntime", "ctranslate2"],
)
lau_pyz = PYZ(lau.pure)
EXE(lau_pyz, lau.scripts, lau.binaries, lau.datas, [], name="Parley", console=False, upx=False, icon=ICON)
