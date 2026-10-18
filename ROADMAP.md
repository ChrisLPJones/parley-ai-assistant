# Roadmap

Where Parley is and where it is going. Plans can change.

## Released

Version 0.1.0 (early release, Windows):

- Push-to-talk voice assistant with local speech recognition, LLM and voice
- Windows installer (per user, no administrator rights, upgrades in place)
- Tray app with state icons, and a Settings window (General and Voice pages)
- Optional Anthropic and OpenAI backends, with keys kept in the Windows credential store
- Tools: read folders and files, find and open things, web search
- Automatic NVIDIA GPU use, with CPU fallback everywhere
- Automatic updates: Parley checks an update server, verifies each file and can roll back

## In progress

- Settings window pages: Models, LLM (including API keys) and Advanced

## Planned

- **Models page:** see the speech, language and voice models available and which ones are in use,
  and upgrade or downgrade them from Settings: download a bigger or smaller model (with progress
  and cancel), switch to it, and delete the ones you no longer need. Each model shows its size
  and whether it suits your hardware.
- **LLM page:** choose the backend and enter API keys in Settings instead of the command line.
- **Advanced page:** the remaining configuration options.
- **Experimental full control mode:** an opt-in setting that lets the assistant control the whole
  PC (run commands, open and close apps, type and click, change and delete files), going well
  beyond today's mostly read-only tools. It will be off by default and clearly marked
  experimental. Turning it on will show a warning that explains the risks and ask you to confirm
  every time it is enabled. It will show an indicator while active, ask before risky actions,
  keep a log of what it did and have an instant way to switch it off.
- **Code signing:** so Windows SmartScreen no longer warns about the installer.
- **Linux support** (Fedora and Ubuntu): an install option, tray icon and GPU support.
- **Wayland support**, after Linux support lands: global hold-to-talk on Wayland desktops (today the key listener needs X11).

## Known limitations

- Windows 10/11 is the only tested platform; Linux is untested.
- Intel GPUs and CPUs are unverified; discrete AMD GPUs are untested. Everything falls back to the CPU.
- Speech recognition defaults to English.
- The local models are small, so answers are simple; a cloud backend gives smarter answers.
- The installer is unsigned.
