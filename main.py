#!/usr/bin/env python3
"""
AfriCaption: Context-preserving translation pipeline entry point.

Main entry point for the AfriCaption pipeline. Delegates to the CLI module
for command-line interface handling.

Usage:
    python main.py [--test] [--models-yaml CONFIG] [--out-dir DIR]
"""

from africaption.cli import main

if __name__ == "__main__":
    main()
