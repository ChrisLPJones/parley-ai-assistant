"""The Settings window (tkinter / ttk with the sv_ttk theme). Pages: General, Voice.

Runs on the UI thread (`ui.py`); other threads reach it only through `open_settings()` / `ui.post`. Every control comes
from a `settings_schema.Setting` (label, help, range, validation) and is checked on each change: an invalid value shows an
inline message and blocks Save. Save writes only the changed keys through `config_io.save_changes` (comments and outside
edits survive), then `app.apply_config(new)` makes them live and the banner says "Applied" or "N settings need a restart".
An API key never appears here: keys live in the credential store and are not on these pages.

`Host` is everything the window needs from the running app (config path, the `Parley` object once it exists, restart,
hotkey capture, device lists, autostart); `ui_preview.py` supplies a stub, tray mode the real one.
"""
import copy
import logging
import threading
import tkinter as tk
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from tkinter import ttk

from . import autostart, config, config_io, hotkey as hotkey_mod, settings_schema, ui, voices

log = logging.getLogger(__name__)

PAGES = (("general", "General"), ("voice", "Voice"))
GENERAL_FIELDS = ("hotkey.key", "audio.input_device", "audio.output_device", "audio.sounds", "@autostart", "updates.check")
VOICE_FIELDS = ("tts.voice", "tts.speed")
SYSTEM_DEFAULT = "System default"
CAPTURE_TIMEOUT_MS = 10_000
POLL_MS = 250
SPEED_STEP = 0.05
MAX_LISTED = 6  # setting names spelled out in a banner


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
                 autostart_api=None, on_applied: Callable[[object], None] | None = None) -> None:
        self.config_path = Path(config_path)
        self._get_app, self._get_error, self._restart = get_app, get_error, restart
        self._get_hotkey = get_hotkey
        self.list_devices = list_devices
        self.autostart = autostart_api or RealAutostart()
        self.on_applied = on_applied

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
        w, h = min(int(e * 40), int(sw * 0.9)), min(int(e * 33), int(sh * 0.9))
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

    def _nav_changed(self, _event=None) -> None:
        sel = self.nav.selection()
        if sel and sel[0] != getattr(self, "page", None):
            self.show_page(sel[0])

    def _title(self, parent, text: str) -> None:
        font = "SunValleySubtitleFont" if self.theme in ("dark", "light") else None
        ttk.Label(parent, text=text, font=font).grid(sticky="w", pady=(0, self.em // 2))

    def _row(self, parent) -> ttk.Frame:
        row = ttk.Frame(parent)
        row.grid(sticky="ew", pady=(0, self.em // 2))
        row.columnconfigure(0, weight=1)
        return row

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
        self.fields[s.path] = Field(s, var.get, err, [cb])

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
        if hasattr(self, "sample_btn"):
            self.sample_btn.state(["!disabled"] if state else ["disabled"])
            if note or self.sample_note.cget("text") in ("Playing or busy...", "Available once Parley has started.", "Starting..."):
                self.sample_note.configure(text=note)
        self.top.after(POLL_MS, self._poll)

    # ---- values and validation ----
    def _base_value(self, path: str):
        table, key = path.split(".")
        return getattr(getattr(self.base, table), key)

    def _changed(self, path: str) -> None:
        self._validate(path)
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
        for path in self.fields:
            valid, value = self._validate(path)
            if not valid:
                ok = False
            elif value != self._base_value(path):
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
        threading.Thread(target=self.host.restart, daemon=True, name="restart").start()

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
        self._after_save(new, bool(changes), autostart_problem)

    def _after_save(self, new, wrote_config: bool, autostart_problem: str | None) -> None:
        app = self.host.app
        prefix = f"Saved, but Start with Windows failed: {autostart_problem} " if autostart_problem else ""
        if not wrote_config:
            return self.show_banner("error" if autostart_problem else "ok", prefix or "Applied.")
        if app is None:
            return self.show_banner("ok", prefix + "Saved. These apply when Parley has started.")
        self.save_btn.state(["disabled"])
        self.show_banner("ok", "Applying...")

        def work() -> None:
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
            ui.post(lambda: self._show_result(kind, prefix + text, bool(pending)))

        threading.Thread(target=work, daemon=True, name="apply-settings").start()

    def _show_result(self, kind: str, text: str, restart_button: bool) -> None:
        if not self.alive():
            return
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
