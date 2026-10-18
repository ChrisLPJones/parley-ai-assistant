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

## In progress

- Settings window pages: Models, LLM (including API keys) and Advanced

## Planned

- **Models page:** download and switch models from Settings.
- **LLM page:** choose the backend and enter API keys in Settings instead of the command line.
- **Advanced page:** the remaining configuration options.
- **Code signing:** so Windows SmartScreen no longer warns about the installer.
- **In-app updates:** an update server so Parley can offer new versions (the update mechanism is built, but no server is live yet).
- **Linux support** (Fedora and Ubuntu): an install option, tray icon and GPU support.

## Known limitations

- Windows 10/11 is the only tested platform; Linux is untested.
- Intel GPUs and CPUs are unverified; discrete AMD GPUs are untested. Everything falls back to the CPU.
- Speech recognition defaults to English.
- The local models are small, so answers are simple; a cloud backend gives smarter answers.
- The installer is unsigned.
