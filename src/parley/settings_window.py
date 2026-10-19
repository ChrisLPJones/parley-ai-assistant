"""The Settings window (tkinter / ttk with the sv_ttk theme). Pages: General, Voice, Models, LLM.

Runs on the UI thread (`ui.py`); other threads reach it only through `open_settings()` / `ui.post`. Every control comes
from a `settings_schema.Setting` (label, help, range, validation) and is checked on each change: an invalid value shows an
inline message and blocks Save. Save writes only the changed keys through `config_io.save_changes` (comments and outside
edits survive), then `app.apply_config(new)` makes them live and the banner says "Applied" or "N settings need a restart".
An API key never goes through Save: the LLM page has its own masked entry with Save key / Remove / Test key buttons that talk
to the credential store only (`Host.keys`), never to config.toml, the settings table or any message. The entry is cleared as
soon as it is used and a stored key is never shown.

`Host` is everything the window needs from the running app (config path, the `Parley` object once it exists, restart,
hotkey capture, device lists, autostart); `ui_preview.py` supplies a stub, tray mode the real one.
"""
import copy
import logging
import os
import threading
import tkinter as tk
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from tkinter import messagebox, ttk

from . import (autostart, config, config_io, credstore, hotkey as hotkey_mod, models, models_view, settings_schema, ui,
               voices)

log = logging.getLogger(__name__)

PAGES = (("general", "General"), ("voice", "Voice"), ("models", "Models"), ("llm", "LLM"))
GENERAL_FIELDS = ("hotkey.key", "audio.input_device", "audio.output_device", "audio.sounds", "@autostart", "updates.check")
VOICE_FIELDS = ("tts.voice", "tts.speed")
SYSTEM_DEFAULT = "System default"
CAPTURE_TIMEOUT_MS = 10_000
POLL_MS = 250
SPEED_STEP = 0.05
MAX_LISTED = 6  # setting names spelled out in a banner
BACKEND_LABELS = {"builtin": "Built-in (runs on this computer)", "ollama": "Ollama (on this computer)",
                  "server": "Your own server (OpenAI-compatible)", "openai": "OpenAI (cloud)", "anthropic": "Anthropic Claude (cloud)"}
PROVIDER_NAMES = {"openai": "OpenAI", "anthropic": "Anthropic"}
ALL_BACKENDS = tuple(BACKEND_LABELS)
LOCAL_SERVERS = ("ollama", "server")
# Which backends each language model setting applies to (a field left out is shown for every backend). Settings of a backend
# that is not selected are hidden and not checked or saved.
LLM_BACKENDS = {
    "llm.model": ("ollama", "server", "openai", "anthropic"), "llm.base_url": ("ollama", "server", "openai"),
    "llm.effort": ("anthropic",), "llm.keep_alive": ("ollama",), "llm.num_ctx": ("builtin", "ollama"),
    "llm.think": ("builtin", "ollama"), "llm.temperature": ("builtin", "ollama", "server"),
}
# Example text for the local backends (shown muted, never saved or applied; no addresses)
LOCAL_EXAMPLES = {
    "builtin": "Presets: tiny, small or medium. Or your own: repo unsloth/Qwen3.5-9B-GGUF with file Qwen3.5-9B-Q4_K_M.gguf (Advanced).",
    "ollama": "Example model name: qwen3:4b",
    "server": "Example model name: Qwen3.5-4B-Q4_K_M",
}
CLOUD_NOTICE = ("Your speech is still turned into text on this computer, but that text, and anything the tools read or "
                "find, is sent to {provider}.")


# ---------- what the window needs from the app ----------

def list_audio_devices(kind: str) -> list[str]:
    """Names of the input or output devices (one host API on Windows, so each device is listed once). Empty on failure."""
    try:
        import sys

        import sounddevice as sd

        mme = next((i for i, h in enumerate(sd.query_hostapis()) if h["name"] == "MME"), None) if sys.platform == "win32" else None
        names: list[str] = []
        for dev in sd.query_devices():
            name = dev["name"]
            if dev[f"max_{kind}_channels"] > 0 and (mme is None or dev["hostapi"] == mme) \
                    and "sound mapper" not in name.lower() and name not in names:
                names.append(name)
        return names
    except Exception as e:
        log.info("Could not list audio devices: %s", e)
        return []


class RealAutostart:
    def status(self) -> autostart.State:
        return autostart.status()

    def set_enabled(self, on: bool) -> None:
        autostart.set_enabled(on)


class Host:
    """The window's view of the app. `get_app()` is the `Parley` object or None while it starts (or failed to)."""

    def __init__(self, config_path: Path, get_app: Callable[[], object] = lambda: None,
                 get_error: Callable[[], str | None] = lambda: None, restart: Callable[[], None] | None = None,
                 get_hotkey: Callable[[], object] = lambda: None, list_devices: Callable[[str], list[str]] = list_audio_devices,
                 autostart_api=None, on_applied: Callable[[object], None] | None = None,
                 models_api=None, detect_hardware: Callable[[str], models.Hardware] | None = None,
                 keys_api=None, key_check: Callable[[str, str], str] = credstore.check_key) -> None:
        self.config_path = Path(config_path)
        self._get_app, self._get_error, self._restart = get_app, get_error, restart
        self._get_hotkey = get_hotkey
        self.list_devices = list_devices
        self.autostart = autostart_api or RealAutostart()
        self.on_applied = on_applied
        self.models = models_api  # a models.ModelManager (or a stub); None = the Models page says it is unavailable
        self._detect_hardware = detect_hardware
        self.keys = keys_api  # a credstore.CredStore (or a fake); None = the LLM page says keys can't be saved here
        self.key_check = key_check  # (backend, key) -> a plain sentence; one request to the provider's own host

    def hardware(self, language: str = "en") -> models.Hardware:
        """Probe this computer (slow: worker thread only)."""
        if self._detect_hardware is not None:
            return self._detect_hardware(language)
        return models.detect_hardware(config.app_dir(), language)

    @property
    def app(self):
        return self._get_app()

    @property
    def error(self) -> str | None:
        return self._get_error()

    def can_restart(self) -> bool:
        return self._restart is not None

    def restart(self) -> None:
        if self._restart is not None:
            self._restart()

    def can_capture(self) -> bool:
        return self._get_hotkey() is not None

    def capture_key(self, callback: Callable[[int], None]) -> bool:
        """The next key press goes to `callback(vk)` (on another thread) and is passed through to other apps."""
        hk = self._get_hotkey()
        if hk is None:
            return False
        hk.capture = callback
        return True

    def cancel_capture(self) -> None:
        hk = self._get_hotkey()
        if hk is not None:
            hk.capture = None


# ---------- pure helpers (tested without Tk) ----------

def label_of(path: str) -> str:
    s = settings_schema.BY_PATH.get(path)
    return s.label if s else path


def summarize(applied: list[str], restart: list[str], deferred: list[str] = ()) -> tuple[str, str]:
    """(kind, banner text) for a save. kind: ok | warn."""
    if restart:
        labels = list(dict.fromkeys(label_of(p) for p in restart))
        names = ", ".join(labels[:MAX_LISTED]) + (f" and {len(labels) - MAX_LISTED} more" if len(labels) > MAX_LISTED else "")
        n = len(restart)
        return "warn", f"Saved. {n} setting{'s need' if n != 1 else ' needs'} a restart ({names})."
    if deferred:
        n = len(deferred)
        return "ok", f"Saved. {n} setting{'s' if n != 1 else ''} will apply after this answer."
    immediate = [p for p in applied if settings_schema.BY_PATH[p].apply != settings_schema.NEXT_LAUNCH]
    if applied and not immediate:
        return "ok", "Saved. This takes effect the next time Parley starts."
    return "ok", "Applied."


def restart_banner(restart: list[str]) -> str:
    kind, text = summarize([], restart)
    return text.removeprefix("Saved. ")


def device_choices(current, names: list[str]) -> tuple[list[str], dict[str, object], str]:
    """(labels for the list, label -> config value, label of the current value). None / "" = the system default."""
    labels = [SYSTEM_DEFAULT, *names]
    mapping: dict[str, object] = {SYSTEM_DEFAULT: None, **{n: n for n in names}}
    if current in (None, ""):
        return labels, mapping, SYSTEM_DEFAULT
    if isinstance(current, str) and current in names:
        return labels, mapping, current
    label = f"Device {current}" if isinstance(current, int) else f"{current} (not found)"
    labels.append(label)
    mapping[label] = current
    return labels, mapping, label


def key_label(name: str) -> str:
    return name.replace("_", " ").title()


@dataclass
class Field:
    setting: settings_schema.Setting
    to_value: Callable[[], object]  # the control's value as a config value; may raise ValueError
    error_label: ttk.Label | None = None
    widgets: list = field(default_factory=list)
    row: ttk.Frame | None = None  # the row to show or hide
    backends: tuple | None = None  # shown (and checked) only for these llm.backend values; None = always


@dataclass
class KeyRow:
    """Widgets of one provider's API key row. The entry holds a key only between typing and pressing Save key."""
    frame: ttk.Frame
    status: ttk.Label
    entry: ttk.Entry
    save: ttk.Button
    remove: ttk.Button
    test: ttk.Button
    message: ttk.Label


# ---------- the window ----------

class SettingsWindow:
    def __init__(self, root: tk.Tk, host: Host, page: str = "general") -> None:
        self.host = host
        self.fields: dict[str, Field] = {}
        self.errors: dict[str, str] = {}
        self.fatal: str | None = None  # why saving is impossible (config.toml has a mistake)
        self.closed = False
        self._capture_job = None
        self._banner_kind = "ok"
        self._explicit_rows: dict[str, int] = {}  # parent widget -> next grid row (see _next_row)
        self._llm_watch = False  # a saved language model change is loading: follow app.llm_state and report the result
        self.top = tk.Toplevel(root)
        self.top.withdraw()
        self.top.title("Parley settings")
        self.theme = ui.apply_theme(root)
        self.dark = self.theme == "dark"
        self.em = ui.em(self.top)
        self._styles()
        self._load_base()
        self.autostart_state = host.autostart.status()
        self._build(page)
        self.top.protocol("WM_DELETE_WINDOW", self.close)
        self.top.bind("<Escape>", lambda e: self.close())
        self.top.bind("<Control-s>", lambda e: self.save())
        self._place()
        ui.set_dark_title_bar(self.top, self.dark)
        self.top.deiconify()
        self.focus()
        self._poll()
        self._initial_banner()

    # ---- setup ----
    def _styles(self) -> None:
        st = ttk.Style(self.top)
        self.muted = "#a6a6a6" if self.dark else "#5f5f5f"
        self.colors = {"ok": "#7ddc8b" if self.dark else "#1b6e2e", "warn": "#ffcc66" if self.dark else "#8a5a00",
                       "error": "#ff8a80" if self.dark else "#b3261e"}
        st.configure("Muted.TLabel", foreground=self.muted)
        st.configure("Error.TLabel", foreground=self.colors["error"])
        st.configure("Treeview", rowheight=int(self.em * 1.6))

    def _load_base(self) -> None:
        try:
            self.base = config.load(self.host.config_path)
        except SystemExit as e:  # a mistake in config.toml: show it, don't offer to overwrite the file
            self.fatal = str(e.code)[:300]
            app = self.host.app
            self.base = copy.deepcopy(app.cfg) if app is not None else config.Config()

    def _place(self) -> None:
        e = self.em
        sw, sh = self.top.winfo_screenwidth(), self.top.winfo_screenheight()
        w, h = min(int(e * 58), int(sw * 0.9)), min(int(e * 42), int(sh * 0.9))
        self.top.minsize(min(int(e * 34), w), min(int(e * 26), h))
        x, y = (sw - w) // 2, max(0, (sh - h) // 3)
        self.top.geometry(f"{w}x{h}+{x}+{y}")

    def focus(self) -> None:
        if self.top.state() == "iconic":
            self.top.deiconify()
        self.top.lift()
        self.top.attributes("-topmost", True)  # Windows only lets a background process raise a window this way
        self.top.after(150, lambda: self.top.attributes("-topmost", False))
        self.top.focus_force()

    def alive(self) -> bool:
        if self.closed:
            return False
        try:
            return bool(self.top.winfo_exists())
        except tk.TclError:
            return False

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        if self._capture_job is not None:
            self.top.after_cancel(self._capture_job)
        self.host.cancel_capture()
        try:
            self.top.destroy()
        except tk.TclError:
            pass
        _forget(self)

    # ---- layout ----
    def _build(self, first_page: str) -> None:
        e = self.em
        pad = max(4, e // 2)
        top = self.top
        top.columnconfigure(1, weight=1)
        top.rowconfigure(0, weight=1)
        # navigation (left)
        self.nav = ttk.Treeview(top, show="tree", selectmode="browse", height=len(PAGES))
        self.nav.column("#0", width=int(e * 8), stretch=False)
        for key, title in PAGES:
            self.nav.insert("", "end", iid=key, text=title)
        self.nav.grid(row=0, column=0, sticky="ns", padx=(pad, 0), pady=pad)
        self.nav.bind("<<TreeviewSelect>>", self._nav_changed)
        # pages (right)
        self.pages: dict[str, ttk.Frame] = {}
        for key, _title in PAGES:
            frame = ttk.Frame(top, padding=(pad * 2, pad, pad, pad))
            frame.columnconfigure(0, weight=1)
            self.pages[key] = frame
            frame.grid(row=0, column=1, sticky="nsew")
            frame.grid_remove()
        self._build_general(self.pages["general"])
        self._build_voice(self.pages["voice"])
        self._build_models(self.pages["models"])
        self._build_llm(self.pages["llm"])
        # banner + buttons (bottom)
        bar = ttk.Frame(top, padding=(pad, 0, pad, pad))
        bar.grid(row=1, column=0, columnspan=2, sticky="ew")
        bar.columnconfigure(0, weight=1)
        self.banner = ttk.Label(bar, text="", wraplength=int(e * 24), anchor="w", justify="left")
        self.banner.grid(row=0, column=0, sticky="ew", padx=(pad, pad))
        bar.bind("<Configure>", lambda ev: self.banner.configure(wraplength=max(int(e * 8), ev.width - int(e * 17))))
        self.restart_btn = ttk.Button(bar, text="Restart now", command=self._restart_now)
        self.restart_btn.grid(row=0, column=1, padx=(0, pad))
        self.restart_btn.grid_remove()
        self.save_btn = ttk.Button(bar, text="Save", style="Accent.TButton" if self.theme in ("dark", "light") else "TButton",
                                   command=self.save)
        self.save_btn.grid(row=0, column=2, padx=(0, pad // 2))
        self.close_btn = ttk.Button(bar, text="Close", command=self.close)
        self.close_btn.grid(row=0, column=3)
        self.show_page(first_page if first_page in self.pages else "general")

    def show_page(self, key: str) -> None:
        for k, frame in self.pages.items():
            frame.grid() if k == key else frame.grid_remove()
        if self.nav.selection() != (key,):
            self.nav.selection_set(key)
        self.page = key
        if key == "models" and getattr(self, "model_tree", None) is not None:
            self._models_refresh()

    def _nav_changed(self, _event=None) -> None:
        sel = self.nav.selection()
        if sel and sel[0] != getattr(self, "page", None):
            self.show_page(sel[0])

    def _title(self, parent, text: str) -> None:
        font = "SunValleySubtitleFont" if self.theme in ("dark", "light") else None
        ttk.Label(parent, text=text, font=font).grid(sticky="w", pady=(0, self.em // 2))

    def _row(self, parent) -> ttk.Frame:
        row = ttk.Frame(parent)
        row.grid(sticky="ew", pady=(0, self.em // 2), **self._next_row(parent))
        row.columnconfigure(0, weight=1)
        return row

    def _next_row(self, parent) -> dict:
        """Grid options for the next row of `parent`. Only for parents that show and hide rows (the LLM tabs): an implicit
        row number would reuse the row of a hidden widget, so those get explicit ones."""
        if str(parent) not in self._explicit_rows:
            return {}
        n = self._explicit_rows[str(parent)]
        self._explicit_rows[str(parent)] = n + 1
        return {"row": n, "column": 0}

    def _help_and_error(self, row, setting, first_row: int, indent: int = 0, show_help: bool = True) -> ttk.Label:
        wrap = int(self.em * 24)
        if show_help:
            ttk.Label(row, text=setting.help, style="Muted.TLabel", wraplength=wrap, justify="left").grid(
                row=first_row, column=0, sticky="w", padx=(indent, 0))
        err = ttk.Label(row, text="", style="Error.TLabel", wraplength=wrap, justify="left")
        err.grid(row=first_row + 1, column=0, sticky="w", padx=(indent, 0))
        err.grid_remove()
        return err

    def _indent(self) -> int:
        return int(self.em * 2.1)  # checkbutton box width, so help text lines up with the label

    # ---- page: General ----
    def _build_general(self, page) -> None:
        self._title(page, "General")
        for path in GENERAL_FIELDS:
            if path == "@autostart":
                self._autostart_row(page)
                continue
            s = settings_schema.BY_PATH[path]
            if s.type == "hotkey":
                self._hotkey_row(page, s)
            elif s.type == "device":
                self._device_row(page, s)
            elif s.type == "bool":
                self._bool_row(page, s)

    def _bool_row(self, page, s) -> None:
        row = self._row(page)
        var = tk.BooleanVar(value=bool(self._base_value(s.path)))
        style = "Switch.TCheckbutton" if self.theme in ("dark", "light") else "TCheckbutton"
        cb = ttk.Checkbutton(row, text=s.label, variable=var, style=style, command=lambda p=s.path: self._changed(p))
        cb.grid(row=0, column=0, sticky="w")
        err = self._help_and_error(row, s, 1, self._indent())
        self.fields[s.path] = Field(s, var.get, err, [cb], row=row)
        return row

    def _hotkey_row(self, page, s) -> None:
        row = self._row(page)
        row.columnconfigure(0, weight=0)
        row.columnconfigure(2, weight=1)
        ttk.Label(row, text=s.label).grid(row=0, column=0, sticky="w", columnspan=2)
        self.key_name = self._base_value(s.path)
        self.key_text = tk.StringVar(value=key_label(self.key_name))
        entry = ttk.Entry(row, textvariable=self.key_text, state="readonly", width=18)
        entry.grid(row=1, column=0, sticky="w")
        self.capture_btn = ttk.Button(row, text="Press a key", command=self._start_capture)
        self.capture_btn.grid(row=1, column=1, sticky="w", padx=(self.em // 2, 0))
        if not self.host.can_capture():
            self.capture_btn.state(["disabled"])
        err = self._help_and_error(row, s, 2)
        for w in [*row.grid_slaves(row=2), *row.grid_slaves(row=3)]:  # help and error text span both columns
            w.grid_configure(columnspan=3)
        self.fields[s.path] = Field(s, lambda: self.key_name, err, [entry, self.capture_btn])

    def _start_capture(self) -> None:
        if self._capture_job is not None:
            return self._end_capture()  # a second click cancels
        done = lambda vk: ui.post(lambda: self._captured(vk))  # noqa: E731 - runs on the key listener's thread
        if not self.host.capture_key(done):
            return
        self.capture_btn.configure(text="Cancel")
        self.key_text.set("Press the key to hold...")
        self._capture_job = self.top.after(CAPTURE_TIMEOUT_MS, self._end_capture)

    def _end_capture(self) -> None:
        if self._capture_job is not None:
            self.top.after_cancel(self._capture_job)
            self._capture_job = None
        self.host.cancel_capture()
        self.capture_btn.configure(text="Press a key")
        self.key_text.set(key_label(self.key_name))

    def _captured(self, vk: int) -> None:
        if self.closed or self._capture_job is None:
            return
        self._end_capture()
        if vk == 0x1B:  # Escape cancels
            return
        name = hotkey_mod.vk_to_name(vk)
        if name is None:
            return self._set_error("hotkey.key", "That key can't be used. Try another, such as Pause, Scroll Lock or F13.")
        self.key_name = name
        self.key_text.set(key_label(name))
        self._changed("hotkey.key")

    # ---- device pickers ----
    def _device_row(self, page, s) -> None:
        kind = "input" if s.path.endswith("input_device") else "output"
        labels, mapping, current = device_choices(self._base_value(s.path), self.host.list_devices(kind))
        row = self._row(page)
        ttk.Label(row, text=s.label).grid(row=0, column=0, sticky="w")
        var = tk.StringVar(value=current)
        box = ttk.Combobox(row, textvariable=var, values=labels, state="readonly", width=44)
        box.grid(row=1, column=0, sticky="ew")
        box.bind("<<ComboboxSelected>>", lambda e, p=s.path: self._changed(p))
        err = self._help_and_error(row, s, 2, show_help=False)  # "System default" is in the box; no need to say it again
        self.fields[s.path] = Field(s, lambda: mapping[var.get()], err, [box])

    # ---- start with Windows ----
    def _autostart_row(self, page) -> None:
        row = self._row(page)
        st = self.autostart_state
        self.autostart_var = tk.BooleanVar(value=st.enabled)
        style = "Switch.TCheckbutton" if self.theme in ("dark", "light") else "TCheckbutton"
        self.autostart_cb = ttk.Checkbutton(row, text="Start with Windows", variable=self.autostart_var, style=style)
        self.autostart_cb.grid(row=0, column=0, sticky="w")
        if not st.available:
            self.autostart_cb.state(["disabled"])
        text = st.reason or "Start Parley in the notification area when you sign in."
        ttk.Label(row, text=text, style="Muted.TLabel", wraplength=int(self.em * 24), justify="left").grid(
            row=1, column=0, sticky="w", padx=(self._indent(), 0))
        self.autostart_error = ttk.Label(row, text="", style="Error.TLabel", wraplength=int(self.em * 24), justify="left")
        self.autostart_error.grid(row=2, column=0, sticky="w", padx=(self._indent(), 0))
        self.autostart_error.grid_remove()

    # ---- page: Voice ----
    def _build_voice(self, page) -> None:
        self._title(page, "Voice")
        page.rowconfigure(2, weight=1)
        s = settings_schema.BY_PATH["tts.voice"]
        speed = settings_schema.BY_PATH["tts.speed"]
        app = self.host.app
        names: list[str] = []
        if app is not None:
            try:
                names = list(app.voices())
            except Exception as e:
                log.info("Could not list voices: %s", e)
        self.voice_names = set(names)
        current = self._base_value("tts.voice")
        if not names:
            names = [current]
        ttk.Label(page, text=s.label).grid(row=1, column=0, sticky="w")
        box = ttk.Frame(page)
        box.grid(row=2, column=0, sticky="nsew")
        box.columnconfigure(0, weight=1)
        box.rowconfigure(0, weight=1)
        self.voice_tree = ttk.Treeview(box, show="tree", selectmode="browse", height=7)
        sb = ttk.Scrollbar(box, orient="vertical", command=self.voice_tree.yview)
        self.voice_tree.configure(yscrollcommand=sb.set)
        self.voice_tree.grid(row=0, column=0, sticky="nsew")
        sb.grid(row=0, column=1, sticky="ns")
        for gi, (group, items) in enumerate(voices.group_voices(names)):
            gid = f"group:{gi}"
            opens = any(v == current for v, _ in items)
            self.voice_tree.insert("", "end", iid=gid, text=group, open=opens)
            for v, label in items:
                self.voice_tree.insert(gid, "end", iid=v, text=label)
        if self.voice_tree.exists(current):
            self.voice_tree.selection_set(current)
            self.top.after(120, self._reveal_voice, current)  # after the first layout, so the list has its final height
        self.voice = current
        self.voice_tree.bind("<<TreeviewSelect>>", self._voice_selected)
        self.voice_help = ttk.Frame(page)
        self.voice_help.grid(row=3, column=0, sticky="ew", pady=(self.em // 4, 0))
        err = self._help_and_error(self.voice_help, s, 0)
        self.fields["tts.voice"] = Field(s, lambda: self.voice, err, [self.voice_tree])
        # speed
        row = self._row(page)
        row.grid(row=4, column=0, sticky="ew", pady=(self.em // 2, 0))
        ttk.Label(row, text=speed.label).grid(row=0, column=0, sticky="w")
        self.speed_var = tk.DoubleVar(value=float(self._base_value("tts.speed")))
        self.speed_text = tk.StringVar(value=self._speed_label())
        scale = ttk.Scale(row, from_=speed.range[0], to=speed.range[1], variable=self.speed_var, command=self._speed_moved)
        scale.grid(row=1, column=0, sticky="ew")
        ttk.Label(row, textvariable=self.speed_text, width=6, anchor="e").grid(row=1, column=1, padx=(self.em // 2, 0))
        err = self._help_and_error(row, speed, 2)
        self.fields["tts.speed"] = Field(speed, lambda: round(self.speed_var.get(), 2), err, [scale])
        # sample
        sample = ttk.Frame(page)
        sample.grid(row=5, column=0, sticky="ew", pady=(self.em // 2, 0))
        sample.columnconfigure(1, weight=1)
        self.sample_btn = ttk.Button(sample, text="Play sample", command=self._play_sample)
        self.sample_btn.grid(row=0, column=0, sticky="w")
        self.sample_note = ttk.Label(sample, text="", style="Muted.TLabel", wraplength=int(self.em * 16), justify="left")
        self.sample_note.grid(row=0, column=1, sticky="w", padx=(self.em // 2, 0))

    def _reveal_voice(self, voice: str) -> None:
        if self.alive() and self.voice_tree.exists(voice):
            self.voice_tree.yview_moveto(0)  # from the top, then scroll only as far as needed to show the voice
            self.voice_tree.see(voice)

    def _speed_label(self) -> str:
        return f"{self.speed_var.get():.2f}x"

    def _speed_moved(self, _value=None) -> None:
        snapped = round(round(self.speed_var.get() / SPEED_STEP) * SPEED_STEP, 2)
        if abs(snapped - self.speed_var.get()) > 1e-9:
            self.speed_var.set(snapped)
        self.speed_text.set(self._speed_label())
        self._changed("tts.speed")

    def _voice_selected(self, _event=None) -> None:
        sel = self.voice_tree.selection()
        if not sel or sel[0].startswith("group:"):
            return  # a heading: keep the voice
        self.voice = sel[0]
        self._changed("tts.voice")

    def _play_sample(self) -> None:
        app = self.host.app
        if app is None:
            return
        voice, speed = self.voice, self.speed_var.get()
        self.sample_btn.state(["disabled"])
        self.sample_note.configure(text="Starting...")

        def work() -> None:
            note = ""
            try:
                if not app.preview_voice(voice, speed, voices.sample_text(voice)):
                    note = "Parley is busy right now."
            except ValueError as e:
                note = str(e)[:100]
            except Exception:
                log.exception("Voice sample failed")
                note = "The sample could not be played."
            ui.post(lambda: self._sample_started(note))

        threading.Thread(target=work, daemon=True, name="sample-click").start()

    def _sample_started(self, note: str) -> None:
        if self.alive():
            self.sample_note.configure(text=note)

    def _poll(self) -> None:
        """Keep 'Play sample' honest: off while Parley starts, is busy, or has no voice list."""
        if not self.alive():
            return
        app = self.host.app
        if app is None:
            state, note = False, "Available once Parley has started."
        else:
            try:
                busy = bool(app.is_busy())
            except Exception:
                busy = False
            state, note = not busy, "Playing or busy..." if busy else ""
        self._models_poll()
        self._llm_poll()
        if hasattr(self, "sample_btn"):
            self.sample_btn.state(["!disabled"] if state else ["disabled"])
            if note or self.sample_note.cget("text") in ("Playing or busy...", "Available once Parley has started.", "Starting..."):
                self.sample_note.configure(text=note)
        self.top.after(POLL_MS, self._poll)

    # ---- page: Models ----
    def _build_models(self, page) -> None:
        self.model_tree = None
        self.model_choice: dict[str, object] = {}  # path -> value set by "Use"; written by Save, like any other setting
        self.hw: models.Hardware | None = None
        self.model_status: dict[str, models.Status] = {}
        self.model_used: set[str] = set()
        self.model_own = None
        self._models_busy = self._models_again = False
        self._dl_sig = None
        self._title(page, "Models")
        api = self.host.models
        if api is None:
            ttk.Label(page, text="The list of models isn't available right now.", style="Muted.TLabel").grid(sticky="w")
            return
        em = self.em
        wrap = int(em * 42)
        page.rowconfigure(2, weight=1)
        self.hw_text = tk.StringVar(value="Checking this computer...")
        ttk.Label(page, textvariable=self.hw_text, style="Muted.TLabel", wraplength=wrap, justify="left").grid(
            row=1, column=0, sticky="w", pady=(0, em // 2))
        box = ttk.Frame(page)
        box.grid(row=2, column=0, sticky="nsew")
        box.columnconfigure(0, weight=1)
        box.rowconfigure(0, weight=1)
        tree = ttk.Treeview(box, columns=("size", "status", "fit"), selectmode="browse", height=12)
        tree.heading("#0", text="Name", anchor="w")
        tree.heading("size", text="Size", anchor="w")
        tree.heading("status", text="Status", anchor="w")
        tree.heading("fit", text="Fit for this computer", anchor="w")
        tree.column("#0", width=int(em * 15), stretch=True)
        tree.column("size", width=int(em * 5), stretch=False)
        tree.column("status", width=int(em * 10), stretch=False)
        tree.column("fit", width=int(em * 12), stretch=True)
        sb = ttk.Scrollbar(box, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=sb.set)
        tree.grid(row=0, column=0, sticky="nsew")
        sb.grid(row=0, column=1, sticky="ns")
        from tkinter import font as tkfont
        self._bold = tkfont.nametofont("TkDefaultFont").copy()
        self._bold.configure(weight="bold")
        tree.tag_configure("inuse", font=self._bold)
        tree.tag_configure("rec", foreground=self.colors["ok"])
        tree.bind("<<TreeviewSelect>>", lambda e: self._models_buttons())
        self.model_tree = tree
        self.model_detail = ttk.Label(page, text="Select a model to see what it does.", wraplength=wrap, justify="left")
        self.model_detail.grid(row=3, column=0, sticky="w", pady=(em // 2, 0))
        prog = ttk.Frame(page)
        prog.grid(row=4, column=0, sticky="ew", pady=(em // 2, 0))
        prog.columnconfigure(0, weight=1)
        self.model_bar = ttk.Progressbar(prog, maximum=100, mode="determinate")
        self.model_bar.grid(row=0, column=0, sticky="ew")
        self.model_bar_text = ttk.Label(prog, text="", width=24, anchor="e")
        self.model_bar_text.grid(row=0, column=1, padx=(em // 2, 0))
        self.model_progress = prog
        prog.grid_remove()
        btns = ttk.Frame(page)
        btns.grid(row=5, column=0, sticky="w", pady=(em // 2, 0))
        self.model_btns = {}
        for i, (key, text, cmd) in enumerate((("download", "Download", self._model_download), ("cancel", "Cancel", self._model_cancel),
                                              ("use", "Use", self._model_use), ("delete", "Delete", self._model_delete))):
            b = ttk.Button(btns, text=text, command=cmd, state="disabled")
            b.grid(row=0, column=i, padx=(0, em // 2))
            self.model_btns[key] = b
        self.model_note = ttk.Label(page, text="", style="Muted.TLabel", wraplength=wrap, justify="left")
        self.model_note.grid(row=6, column=0, sticky="w", pady=(em // 2, 0))
        self.model_msg = ttk.Label(page, text="", wraplength=wrap, justify="left")
        self.model_msg.grid(row=7, column=0, sticky="w")
        self._dl_sig = self._dl_signature()
        self._models_refresh()

    def _dl_signature(self):
        st = self.host.models.state
        return (st.model_id, st.phase, st.done)

    def _models_say(self, text: str, kind: str = "") -> None:
        self.model_msg.configure(text=text, foreground=self.colors.get(kind, ""))

    def _models_refresh(self) -> None:
        """Scan the models folder (and probe the hardware once) on a worker thread, then redraw."""
        if self.model_tree is None:
            return
        if self._models_busy:
            self._models_again = True
            return
        self._models_busy = True
        api, host, cfg, hw = self.host.models, self.host, self.base, self.hw

        def work() -> None:
            try:
                status, used = api.scan(), api.in_use_ids()
                own = models.own_llm_file(cfg, api.home)
                hw2 = hw or host.hardware(cfg.stt.language or "en")
            except Exception:
                log.exception("Could not list the models")
                status, used, own, hw2 = None, set(), None, hw
            ui.post(lambda: self._models_loaded(status, used, own, hw2))

        threading.Thread(target=work, daemon=True, name="models-scan").start()

    def _models_loaded(self, status, used, own, hw) -> None:
        if not self.alive():
            return
        self._models_busy = False
        if status is None:
            self._models_say("Parley couldn't read the models folder.", "error")
        else:
            self.model_status, self.model_used, self.model_own = status, used, own
            self.hw = hw
            self.hw_text.set(models_view.hardware_line(hw) if hw else "")
            self._models_fill()
        if self._models_again:
            self._models_again = False
            self._models_refresh()

    CHOICE_PATH = {"llm": "llm.preset", "stt": "stt.model", "tts": "tts.model"}  # the setting "Use" sets, per kind

    def _chosen_key(self, kind: str = "llm") -> tuple[str | None, str]:
        """(catalog key the user chose for this kind, "unsaved" | "saved"); None when nothing is chosen (auto, or a custom
        language model)."""
        path = self.CHOICE_PATH[kind]
        saved = self._base_value(path)
        if path in self.model_choice:
            key = self.model_choice[path]
            return key, ("saved" if key == saved else "unsaved")
        if kind == "llm":
            llm = self.base.llm
            if llm.preset != "auto" and not (llm.model_repo or llm.model_path):
                return llm.preset, "saved"
            return None, ""
        return (None, "") if saved == "auto" else (saved, "saved")

    def _models_fill(self) -> None:
        tree = getattr(self, "model_tree", None)
        if tree is None or not self.model_status:
            return
        keep = tree.selection()
        tree.delete(*tree.get_children())
        api = self.host.models
        rec = models.recommended_ids(self.hw, self._chosen_key("tts")[0] or "default") if self.hw else set()
        for kind, title in models_view.GROUPS:
            chosen, how = self._chosen_key(kind)
            gid = f"group:{kind}"
            tree.insert("", "end", iid=gid, text=title, open=True)
            if kind == "llm" and self.model_own is not None:
                o = self.model_own
                tree.insert(gid, "end", iid=models_view.OWN_ID, text=o.label, tags=("inuse",),
                            values=(models_view.format_size(o.bytes), "In use", "Not checked"))
            for m in models_view.ordered(kind, api.catalog):
                st = self.model_status.get(m.id, models.Status("not_installed"))
                in_use = m.id in self.model_used
                pick = how if (chosen == m.key and not in_use) else ""
                level, words = models.suitability(m, self.hw) if self.hw else ("", "Checking...")
                size = st.bytes if st.state == "installed" else m.total_bytes
                tags = (["inuse"] if in_use else []) + (["rec"] if m.id in rec else [])
                tree.insert(gid, "end", iid=m.id, tags=tags, text=m.label + (" - recommended" if m.id in rec else ""),
                            values=(models_view.format_size(size), models_view.status_text(m, st, in_use, pick),
                                    models_view.fit_text(level, words)))
        if keep and tree.exists(keep[0]):
            tree.selection_set(keep[0])
        self._models_buttons()

    def _selected_model(self) -> models.Model | None:
        sel = self.model_tree.selection()
        return self.host.models.catalog.get(sel[0]) if sel else None

    def _models_buttons(self) -> None:
        tree = getattr(self, "model_tree", None)
        if tree is None:
            return
        m = self._selected_model()
        sel = tree.selection()
        if m is None:
            text = ("Your own model file. Choose a built-in model to replace it." if sel and sel[0] == models_view.OWN_ID
                    else "Select a model to see what it does.")
            a = models_view.Actions(False, False, False, False)
            self.model_detail.configure(text=text)
        else:
            st = self.model_status.get(m.id, models.Status("not_installed"))
            chosen, _how = self._chosen_key(m.kind)
            a = models_view.actions(m, st, in_use=m.id in self.model_used, selected=chosen == m.key,
                                    other_chosen=chosen is not None and chosen != m.key, dl=self.host.models.state)
            detail = m.note
            if self.hw:
                level, words = models.suitability(m, self.hw)
                detail += f" {models_view.fit_text(level, words)}. {models_view.warning_text(level)}"
            self.model_detail.configure(text=detail.rstrip())
        for key, b in self.model_btns.items():
            b.state(["!disabled"] if getattr(a, key) else ["disabled"])
        note = a.note
        if m is not None and a.use:
            note = models_view.RESTART_NOTES[m.kind]
            if m.kind == "llm" and self.model_own is not None:  # Use clears the custom repo / file / path, so say so before Save
                note += " " + models_view.OWN_REPLACE_NOTE
        self.model_note.configure(text=note)

    def _models_poll(self) -> None:
        """Called by _poll every POLL_MS: follow the download state (the worker thread never touches Tk)."""
        if getattr(self, "model_tree", None) is None:
            return
        st = self.host.models.state
        sig = (st.model_id, st.phase, st.done)
        running = st.phase in ("downloading", "verifying")
        if running:
            self.model_progress.grid()
            self.model_bar.configure(value=models_view.percent(st.done, st.total))
            self.model_bar_text.configure(text=models_view.progress_text(st))
            if self.model_tree.exists(st.model_id):
                self.model_tree.set(st.model_id, "status", f"Downloading {models_view.percent(st.done, st.total)}%")
        else:
            self.model_progress.grid_remove()
        if sig == self._dl_sig:
            return
        before = self._dl_sig
        self._dl_sig = sig
        if before is None or before[:2] != sig[:2]:  # started, finished, failed or cancelled: redraw and say how it went
            if st.phase in ("error", "cancelled", "done"):
                self._models_say(st.message, "error" if st.phase == "error" else "ok")
            self._models_refresh()

    def _model_download(self) -> None:
        m = self._selected_model()
        if m is None:
            return
        try:
            self.host.models.download(m.id)  # a catalog id only: the manager rebuilds every URL and path itself
        except models.ModelError as e:
            return self._models_say(str(e), "error")
        self._models_say("")
        self._models_poll()
        self._models_refresh()

    def _model_cancel(self) -> None:
        self.host.models.cancel()

    def _model_use(self) -> None:
        m = self._selected_model()
        if m is None or m.kind not in self.CHOICE_PATH:
            return
        if m.kind == "llm":  # a built-in preset replaces any custom repo / file / path (they would win over the preset)
            self.model_choice.update({"llm.preset": m.key, "llm.model_repo": "", "llm.model_file": "", "llm.model_path": ""})
        else:  # m.key is the catalog's Whisper name / Kokoro variant, i.e. already a valid stt.model / tts.model value
            self.model_choice[self.CHOICE_PATH[m.kind]] = m.key
        self._models_say("")
        self._models_fill()

    def _model_delete(self) -> None:
        m = self._selected_model()
        if m is None:
            return
        st = self.model_status.get(m.id, models.Status("not_installed"))
        if not messagebox.askyesno("Delete model", f"Delete {m.label}?\nThis frees {models_view.format_size(st.bytes)}. "
                                   "You can download it again later.", parent=self.top):
            return
        try:
            freed = self.host.models.delete(m.id)
        except models.ModelError as e:
            return self._models_say(str(e), "error")
        self._models_say(f"{m.label} deleted. {models_view.format_size(freed)} freed.", "ok")
        self._models_refresh()

    # ---- page: LLM ----
    def _build_llm(self, page) -> None:
        self._title(page, "Language model")
        e = self.em
        self.keys: dict[str, KeyRow] = {}
        self._backend_rows: list[tuple[ttk.Frame, tuple]] = []  # (row, backends it is shown for)
        notebook = ttk.Notebook(page)
        notebook.grid(row=1, column=0, sticky="nsew")
        page.rowconfigure(1, weight=1)
        model_tab, answers_tab = ttk.Frame(notebook, padding=(0, e // 2)), ttk.Frame(notebook, padding=(0, e // 2))
        for tab in (model_tab, answers_tab):
            tab.columnconfigure(0, weight=1)
            self._explicit_rows[str(tab)] = 0
        notebook.add(model_tab, text="Model")
        notebook.add(answers_tab, text="Behavior")
        # -- backend
        s = settings_schema.BY_PATH["llm.backend"]
        row = self._row(model_tab)
        ttk.Label(row, text=s.label).grid(row=0, column=0, sticky="w")
        self.backend_var = tk.StringVar(value=BACKEND_LABELS.get(self._base_value("llm.backend"), self._base_value("llm.backend")))
        box = ttk.Combobox(row, textvariable=self.backend_var, values=[BACKEND_LABELS[b] for b in ALL_BACKENDS], state="readonly", width=40)
        box.grid(row=1, column=0, sticky="w")
        box.bind("<<ComboboxSelected>>", lambda ev: self._backend_changed())
        err = self._help_and_error(row, s, 2)
        by_label = {v: k for k, v in BACKEND_LABELS.items()}
        self.fields["llm.backend"] = Field(s, lambda: by_label.get(self.backend_var.get(), self.backend_var.get()), err, [box], row=row)
        wrap = int(e * 30)
        self.llm_notice = ttk.Label(model_tab, text="", wraplength=wrap, justify="left")
        self.llm_notice.grid(sticky="w", pady=(0, e // 2), **self._next_row(model_tab))
        hint = ttk.Label(model_tab, text="Choose which built-in model to use on the Models page.", style="Muted.TLabel",
                         wraplength=wrap, justify="left")
        hint.grid(sticky="w", pady=(0, e // 2), **self._next_row(model_tab))
        self._backend_rows.append((hint, ("builtin",)))
        # -- model name, address (+ warning), key, effort, ...
        self._entry_row(model_tab, "llm.model")
        self.llm_example = ttk.Label(model_tab, text="", style="Muted.TLabel", wraplength=wrap, justify="left")
        self.llm_example.grid(sticky="w", pady=(0, e // 2), **self._next_row(model_tab))
        self._entry_row(model_tab, "llm.base_url")
        self.url_warning = ttk.Label(model_tab, text="", wraplength=wrap, justify="left", foreground=self.colors["warn"])
        self.url_warning.grid(sticky="w", pady=(0, e // 2), **self._next_row(model_tab))
        self.url_warning.grid_remove()
        for backend in PROVIDER_NAMES:
            self._key_row(model_tab, backend)
        self._choice_row(model_tab, "llm.effort")
        self._entry_row(model_tab, "llm.keep_alive")
        self._entry_row(model_tab, "llm.num_ctx")
        think_row = self._bool_row(model_tab, settings_schema.BY_PATH["llm.think"])
        self.fields["llm.think"].backends = LLM_BACKENDS["llm.think"]
        self._backend_rows.append((think_row, LLM_BACKENDS["llm.think"]))
        # -- answers
        self._bool_row(answers_tab, settings_schema.BY_PATH["tools.enabled"])
        self._bool_row(answers_tab, settings_schema.BY_PATH["tools.web_search"])
        self._prompt_row(answers_tab)
        self._entry_row(answers_tab, "llm.temperature")
        self._entry_row(answers_tab, "llm.history_turns")
        self._backend_changed(initial=True)

    def _backend(self) -> str:
        return self.fields["llm.backend"].to_value()

    def _active(self, f: Field) -> bool:
        return f.backends is None or self._backend() in f.backends

    def _entry_row(self, parent, path: str) -> None:
        s = settings_schema.BY_PATH[path]
        cur = self._base_value(path)
        kind = {"int": int, "float": float}.get(s.type, str)
        row = self._row(parent)
        ttk.Label(row, text=s.label).grid(row=0, column=0, sticky="w")
        var = tk.StringVar(value=str(cur))
        entry = ttk.Entry(row, textvariable=var, width=44 if kind is str else 12)
        entry.grid(row=1, column=0, sticky="w")
        err = self._help_and_error(row, s, 2)

        def to_value():
            text = var.get().strip() if kind is not str else var.get()
            if kind is str:
                return text
            try:
                return kind(text)
            except ValueError:
                raise ValueError(f"{s.label}: must be a {'whole number' if kind is int else 'number'}") from None

        var.trace_add("write", lambda *a, p=path: self._changed(p))
        backends = LLM_BACKENDS.get(path)
        self.fields[path] = Field(s, to_value, err, [entry], row=row, backends=backends)
        if backends:
            self._backend_rows.append((row, backends))

    def _choice_row(self, parent, path: str) -> None:
        s = settings_schema.BY_PATH[path]
        row = self._row(parent)
        ttk.Label(row, text=s.label).grid(row=0, column=0, sticky="w")
        var = tk.StringVar(value=self._base_value(path))
        box = ttk.Combobox(row, textvariable=var, values=list(s.choices), state="readonly", width=12)
        box.grid(row=1, column=0, sticky="w")
        box.bind("<<ComboboxSelected>>", lambda ev, p=path: self._changed(p))
        err = self._help_and_error(row, s, 2)
        backends = LLM_BACKENDS.get(path)
        self.fields[path] = Field(s, var.get, err, [box], row=row, backends=backends)
        if backends:
            self._backend_rows.append((row, backends))

    def _prompt_row(self, parent) -> None:
        s = settings_schema.BY_PATH["llm.system_prompt"]
        row = self._row(parent)
        ttk.Label(row, text=s.label).grid(row=0, column=0, sticky="w")
        st = ttk.Style(self.top)
        bg = st.lookup("TEntry", "fieldbackground") or ("#2b2b2b" if self.dark else "#ffffff")
        fg = st.lookup("TEntry", "foreground") or ("#fafafa" if self.dark else "#1c1c1c")
        text = tk.Text(row, height=6, width=48, wrap="word", bg=bg, fg=fg, insertbackground=fg, relief="flat", highlightthickness=1,
                       highlightbackground=self.muted, undo=True, font="TkDefaultFont")
        text.insert("1.0", self._base_value("llm.system_prompt"))
        text.edit_modified(False)
        text.grid(row=1, column=0, sticky="ew")

        def modified(_event=None) -> None:
            if text.edit_modified():
                text.edit_modified(False)
                self._changed("llm.system_prompt")

        text.bind("<<Modified>>", modified)
        reset = ttk.Button(row, text="Use the default prompt", command=lambda: (text.delete("1.0", "end"), text.insert(
            "1.0", config.DEFAULT_SYSTEM_PROMPT)))
        reset.grid(row=2, column=0, sticky="w", pady=(self.em // 4, 0))
        err = self._help_and_error(row, s, 3)
        self.fields["llm.system_prompt"] = Field(s, lambda: text.get("1.0", "end-1c"), err, [text, reset], row=row)

    # -- API keys: credential store only
    def _key_row(self, parent, backend: str) -> None:
        e, name = self.em, PROVIDER_NAMES[backend]
        row = self._row(parent)
        row.columnconfigure(0, weight=0)
        ttk.Label(row, text=f"{name} API key").grid(row=0, column=0, columnspan=4, sticky="w")
        wrap = int(e * 30)
        status = ttk.Label(row, text="", wraplength=wrap, justify="left", style="Muted.TLabel")
        status.grid(row=1, column=0, columnspan=4, sticky="w")
        entry = ttk.Entry(row, show="*", width=40)  # masked; no textvariable, never prefilled
        entry.grid(row=2, column=0, columnspan=4, sticky="w", pady=(e // 4, 0))
        save = ttk.Button(row, text="Save key", command=lambda b=backend: self._key_save(b))
        remove = ttk.Button(row, text="Remove", command=lambda b=backend: self._key_remove(b))
        test = ttk.Button(row, text="Test key", command=lambda b=backend: self._key_test(b))
        for i, b in enumerate((save, remove, test)):
            b.grid(row=3, column=i, sticky="w", padx=(0, e // 2), pady=(e // 4, 0))
        message = ttk.Label(row, text="", wraplength=wrap, justify="left")
        message.grid(row=4, column=0, columnspan=4, sticky="w")
        row.columnconfigure(3, weight=1)
        self.keys[backend] = KeyRow(row, status, entry, save, remove, test, message)
        self._backend_rows.append((row, (backend,)))
        self._key_refresh(backend)

    def _key_say(self, backend: str, text: str, kind: str = "") -> None:
        self.keys[backend].message.configure(text=text, foreground=self.colors.get(kind, ""))

    def _key_state(self, backend: str) -> tuple[bool, bool, bool, str]:
        """(saved, usable, can_save, status sentence). Only presence is read: no key value is ever displayed."""
        store, env = self.host.keys, credstore.ENV_VARS[backend]
        if store is None or not store.available:
            return False, bool(os.environ.get(env)), False, (
                f"Saving keys needs the Windows credential store, which isn't available here. Set {env} instead.")
        try:
            saved = store.exists(backend)
        except credstore.CredentialError:
            return False, False, True, "Parley couldn't read the Windows credential store."
        if saved:
            ignored = f" {env} in the environment is ignored." if os.environ.get(env) else ""
            return True, True, True, "Saved in the Windows credential store." + ignored
        if os.environ.get(env):
            return False, True, True, f"No key saved. Parley is using {env} from the environment."
        if self.base.llm.api_key:
            return False, True, True, "A key is still in config.toml. Save it here, then delete that line from the file."
        return False, False, True, "No key saved."

    def _key_refresh(self, backend: str) -> None:
        k = self.keys[backend]
        saved, usable, can_save, text = self._key_state(backend)
        k.status.configure(text=text)
        k.save.configure(text="Replace key" if saved else "Save key")
        k.save.state(["!disabled"] if can_save else ["disabled"])
        k.entry.state(["!disabled"] if can_save else ["disabled"])
        k.remove.state(["!disabled"] if saved else ["disabled"])
        k.test.state(["!disabled"] if usable else ["disabled"])

    def _key_save(self, backend: str) -> None:
        k = self.keys[backend]
        text = k.entry.get().strip()
        k.entry.delete(0, "end")  # cleared at once, whatever happens next
        if not text:
            return self._key_say(backend, "Type or paste the key first.", "error")
        if not credstore.check_key_shape(text):
            return self._key_say(backend, "That doesn't look like an API key.", "error")
        try:
            credstore.replace_key(self.host.keys, backend, text)
        except credstore.CredentialError as e:
            self._key_refresh(backend)
            return self._key_say(backend, str(e)[:200], "error")
        except Exception as e:
            log.warning("Saving the %s key failed (%s)", backend, type(e).__name__)  # the type only, never the text
            return self._key_say(backend, "The key could not be saved.", "error")
        finally:
            text = ""
        self._key_refresh(backend)
        app = self.host.app
        if app is not None and getattr(app.cfg.llm, "backend", None) == backend:
            self._key_say(backend, "Key saved. Loading it now...", "ok")
            self._llm_watch = True
            threading.Thread(target=app.key_changed, args=(backend,), daemon=True, name="key-changed").start()
        else:
            self._key_say(backend, "Key saved.", "ok")

    def _key_remove(self, backend: str) -> None:
        if not messagebox.askyesno("Remove key", f"Remove the saved {PROVIDER_NAMES[backend]} key from the Windows credential store?",
                                   parent=self.top):
            return
        try:
            self.host.keys.delete(backend)
        except credstore.CredentialError as e:
            return self._key_say(backend, str(e)[:200], "error")
        self._key_refresh(backend)
        app = self.host.app
        running = app is not None and getattr(app.cfg.llm, "backend", None) == backend
        self._key_say(backend, "Key removed." + (" Parley keeps using it until you restart." if running else ""), "ok")

    def _key_test(self, backend: str) -> None:
        k = self.keys[backend]
        k.test.state(["disabled"])
        self._key_say(backend, "Asking " + PROVIDER_NAMES[backend] + "...")
        store = self.host.keys or credstore.CredStore(credstore.UnavailableBackend())  # never the real store by accident
        cfg, check = self.base.llm, self.host.key_check

        def work() -> None:
            try:
                key, _source = credstore.lookup_api_key(backend, cfg, store)
                sentence = check(backend, key)
                del key
            except Exception as e:
                log.warning("Key test failed (%s)", type(e).__name__)
                sentence = "The key could not be tested."
            ui.post(lambda: self._key_tested(backend, sentence))

        threading.Thread(target=work, daemon=True, name="key-test").start()

    def _key_tested(self, backend: str, sentence: str) -> None:
        if self.alive():
            self._key_say(backend, sentence)
            self._key_refresh(backend)

    # -- backend switch and warnings
    def _backend_changed(self, initial: bool = False) -> None:
        backend = self._backend()
        for row, backends in self._backend_rows:
            row.grid() if backend in backends else row.grid_remove()
        for path, f in self.fields.items():
            if f.backends is None:
                continue
            if self._active(f):
                f.row.grid()
                if not initial:
                    self._validate(path)
            else:
                f.row.grid_remove()
                self.errors.pop(path, None)
        provider = PROVIDER_NAMES.get(backend)
        self.llm_notice.configure(text=CLOUD_NOTICE.format(provider=provider) if provider else "",
                                  foreground=self.colors["warn"])
        self.llm_notice.grid() if provider else self.llm_notice.grid_remove()
        example = LOCAL_EXAMPLES.get(backend, "")
        self.llm_example.configure(text=example)
        self.llm_example.grid() if example else self.llm_example.grid_remove()
        self._url_warning()
        if not initial:
            self._refresh_buttons()

    def _url_warning(self) -> None:
        backend = self._backend()
        text = None
        if self.fields["llm.base_url"].backends and backend in self.fields["llm.base_url"].backends:
            url = self.fields["llm.base_url"].to_value().strip()
            if url:
                text = credstore.endpoint_warning(backend, url, loopback_ok=backend in LOCAL_SERVERS, host_check=backend in PROVIDER_NAMES,
                                         show_host=False)
        self.url_warning.configure(text=text or "")
        self.url_warning.grid() if text else self.url_warning.grid_remove()

    def _llm_poll(self) -> None:
        """Follow a language model load that Save (or a key change) started and say how it went."""
        if not self._llm_watch:
            return
        app = self.host.app
        state = getattr(app, "llm_state", None) if app is not None else None
        if not state:
            return
        kind, detail = state[0], state[1]
        if kind in ("loading", "pending") and self._banner_kind != "error":
            if not self.banner.cget("text").startswith("Saved. Loading") and not getattr(app, "deferred", []):
                self.show_banner("ok", "Saved. Loading the language model (this can take a minute)...")
        elif kind == "done":
            self._llm_watch = False
            self._models_refresh()  # the model that was chosen is now the one in use
            pending = list(app.restart_pending()) if hasattr(app, "restart_pending") else []
            text = "Saved. The new language model is ready." + (" " + restart_banner(pending) if pending else "")
            self.show_banner("warn" if pending else "ok", text, restart_button=bool(pending))
        elif kind == "error":
            self._llm_watch = False
            self.show_banner("error", f"Saved, but the new language model could not start. {detail} Parley is still using the previous one.",
                             restart_button=True)

    # ---- values and validation ----
    def _base_value(self, path: str):
        table, key = path.split(".")
        return getattr(getattr(self.base, table), key)

    def _changed(self, path: str) -> None:
        self._validate(path)
        if path == "llm.base_url":
            self._url_warning()
        self._refresh_buttons()

    def _set_error(self, path: str, text: str) -> None:
        f = self.fields[path]
        if text:
            self.errors[path] = text
            f.error_label.configure(text=text)
            f.error_label.grid()
        else:
            self.errors.pop(path, None)
            f.error_label.configure(text="")
            f.error_label.grid_remove()
        self._refresh_buttons()

    def _validate(self, path: str) -> tuple[bool, object]:
        """(True, the validated value) or (False, None) after showing the problem next to the control. The value can
        itself be None (a device picker on 'System default')."""
        f = self.fields[path]
        try:
            value = f.setting.validate(f.to_value())
            if path == "tts.voice" and self.voice_names and value != self._base_value(path) and value not in self.voice_names:
                raise ValueError("That voice isn't installed.")
        except ValueError as e:
            self._set_error(path, str(e))
            return False, None
        except Exception:  # a control that can't give a value (never expected): treat as invalid, never crash the window
            log.exception("Could not read %s", path)
            self._set_error(path, "This value can't be read.")
            return False, None
        self._set_error(path, "")
        return True, value

    def collect(self) -> tuple[dict[str, object], bool]:
        """({changed path: value}, all valid). Only values that differ from config.toml are changes."""
        changes, ok = {}, True
        for path, f in self.fields.items():
            if not self._active(f):  # a setting of a backend that is not selected: not checked, not saved
                continue
            valid, value = self._validate(path)
            if not valid:
                ok = False
            elif value != self._base_value(path):
                changes[path] = value
        for path, value in self.model_choice.items():  # "Use" on the Models page (values from the catalog)
            if value != self._base_value(path):
                changes[path] = value
        voice = changes.get("tts.voice")
        if voice and (lang := settings_schema.voice_lang(voice)) and lang != self._base_value("tts.lang"):
            changes["tts.lang"] = lang  # the voice's language follows it
        return changes, ok

    def _autostart_change(self) -> bool | None:
        if not self.autostart_state.available:
            return None
        want = bool(self.autostart_var.get())
        return want if want != self.autostart_state.enabled else None

    def _refresh_buttons(self) -> None:
        blocked = bool(self.errors) or self.fatal is not None
        self.save_btn.state(["disabled"] if blocked else ["!disabled"])

    # ---- banner ----
    def show_banner(self, kind: str, text: str, restart_button: bool = False) -> None:
        self._banner_kind = kind
        self.banner.configure(text=text, foreground=self.colors.get(kind, ""))
        if restart_button and self.host.can_restart():
            self.restart_btn.grid()
        else:
            self.restart_btn.grid_remove()

    def _initial_banner(self) -> None:
        self._refresh_buttons()
        if self.fatal:
            return self.show_banner("error", self.fatal)
        err = self.host.error
        if err:
            return self.show_banner("error", f"Parley could not start: {err}")
        app = self.host.app
        pending = list(app.restart_pending()) if app is not None and hasattr(app, "restart_pending") else []
        if pending:
            self.show_banner("warn", restart_banner(pending), restart_button=True)

    def _restart_now(self) -> None:
        self.restart_btn.state(["disabled"])
        self.show_banner("ok", "Restarting...")
        def work() -> None:
            try:
                self.host.restart()
            except Exception:
                log.exception("Restart failed")
                ui.post(self._restart_failed)

        threading.Thread(target=work, daemon=True, name="restart").start()

    def _restart_failed(self) -> None:
        if self.alive():
            self.restart_btn.state(["!disabled"])
            self.show_banner("error", "Parley couldn't restart. Close it from the tray menu and start it again.", restart_button=True)

    # ---- Save ----
    def save(self) -> None:
        if self.fatal is not None:
            return self.show_banner("error", self.fatal)
        changes, ok = self.collect()
        if not ok:
            return self.show_banner("error", "Fix the highlighted settings, then save.")
        auto = self._autostart_change()
        if not changes and auto is None:
            return self.show_banner("ok", "Nothing to change.")
        if changes:
            try:
                config_io.save_changes(self.host.config_path, changes)
            except config_io.ConfigFileError as e:
                return self.show_banner("error", str(e)[:300])
            except OSError as e:
                return self.show_banner("error", f"Parley couldn't write config.toml ({e.strerror or type(e).__name__}). "
                                        "Check that the file isn't read-only or open in another program.")
            except (TypeError, ValueError):
                log.exception("Saving settings failed")
                return self.show_banner("error", "Those settings couldn't be saved. The log has details.")
        autostart_problem = None
        if auto is not None:
            try:
                self.host.autostart.set_enabled(auto)
                self.autostart_state = self.host.autostart.status()
                self.autostart_error.grid_remove()
            except (OSError, ValueError) as e:
                autostart_problem = str(e)[:200] or "Windows refused the change."
                self.autostart_error.configure(text=autostart_problem)
                self.autostart_error.grid()
        new = copy.deepcopy(self.base)
        for path, value in changes.items():
            table, key = path.split(".")
            setattr(getattr(new, table), key, value)
        self.base = new
        self._models_fill()  # "Chosen (press Save)" becomes "Chosen (restart Parley)"
        self._after_save(new, bool(changes), autostart_problem)

    def _after_save(self, new, wrote_config: bool, autostart_problem: str | None) -> None:
        app = self.host.app
        prefix = f"Saved, but Start with Windows failed: {autostart_problem} " if autostart_problem else ""
        if not wrote_config:
            return self.show_banner("error" if autostart_problem else "ok", prefix or "Applied.")
        if app is None:
            if self.host.can_restart():  # the running copy already read its config; it will not see this save
                return self.show_banner("warn", prefix + "Saved. Restart Parley to use it.", restart_button=True)
            return self.show_banner("ok", prefix + "Saved. These apply when Parley has started.")
        self.save_btn.state(["disabled"])
        self.show_banner("ok", "Applying...")

        def work() -> None:
            deferred: list[str] = []
            try:
                applied, restart = app.apply_config(new)
                deferred = list(getattr(app, "deferred", []))
                kind, text = summarize(applied, restart, deferred)
                if self.host.on_applied:
                    self.host.on_applied(new)
            except Exception:
                log.exception("Applying the settings failed")
                applied, restart, kind, text = [], [], "warn", "Saved, but some changes could not be applied. Restart Parley to use them."
            pending = restart or (list(app.restart_pending()) if hasattr(app, "restart_pending") else [])
            if pending and not restart:
                kind, text = "warn", text + " " + restart_banner(pending)
            loading = getattr(app, "llm_state", ("idle",))[0] in ("pending", "loading")
            if loading and not deferred and not pending:
                kind, text = "ok", "Saved. Loading the language model (this can take a minute)..."
            ui.post(lambda: self._show_result(kind, prefix + text, bool(pending), loading))

        threading.Thread(target=work, daemon=True, name="apply-settings").start()

    def _show_result(self, kind: str, text: str, restart_button: bool, loading: bool = False) -> None:
        if not self.alive():
            return
        self._llm_watch = loading
        self._refresh_buttons()
        self.show_banner("warn" if restart_button and kind == "ok" else kind, text, restart_button)


# ---------- entry points ----------

_current: SettingsWindow | None = None  # UI thread only


def _forget(window: "SettingsWindow | None" = None) -> None:
    """Drop the reference to the closed window; its Tk variables are then freed on the UI thread (see ui.cleanup_hooks)."""
    global _current
    if window is None or _current is window:
        _current = None
    if window is not None:
        window.fields.clear()
        window.__dict__.clear()
        window.closed = True


ui.cleanup_hooks.append(_forget)


def _open(host: Host, page: str | None) -> None:
    global _current
    if _current is not None and _current.alive():
        _current.focus()
        if page:
            _current.show_page(page)
        return
    _current = SettingsWindow(ui.root(), host, page or "general")
    log.info("Settings window opened")


_open_queued = threading.Event()  # a request is already waiting: a flood of "open settings" messages collapses into one


def open_settings(host: Host, page: str | None = None) -> bool:
    """Open the window, or bring the open one to the front (safe to call from any thread, any number of times).
    False when the UI could not start (no display)."""
    if _open_queued.is_set():
        return True

    def run() -> None:
        _open_queued.clear()
        _open(host, page)

    _open_queued.set()
    if ui.post(run):
        return True
    _open_queued.clear()
    return False


def close_settings() -> None:
    """Close the window if it is open (any thread)."""
    def close() -> None:
        if _current is not None and _current.alive():
            _current.close()
    ui.post(close)
