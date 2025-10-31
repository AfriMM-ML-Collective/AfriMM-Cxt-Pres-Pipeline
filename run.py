#!/usr/bin/env python3
"""
Full run script - processes entire dataset.

This script provides a convenient entry point for running the complete
pipeline on the entire AfriMMD dataset.
"""
import subprocess
import sys

if __name__ == "__main__":
    print("="*60)
    print("FULL RUN: Processing entire dataset")
    print("="*60)
    sys.exit(subprocess.run([sys.executable, "main.py"]).returncode)

