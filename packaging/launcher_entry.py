"""PyInstaller entry for Parley.exe: the tiny stdlib-only launcher (src/parley/launcher.py)."""
import sys

from parley.launcher import main

if __name__ == "__main__":
    sys.exit(main())
