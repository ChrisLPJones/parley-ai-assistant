"""PyInstaller entry for parley-app.exe (the app itself; Parley.exe, the launcher, starts it)."""
import multiprocessing
import sys

from parley.__main__ import main

if __name__ == "__main__":
    multiprocessing.freeze_support()
    sys.exit(main())
