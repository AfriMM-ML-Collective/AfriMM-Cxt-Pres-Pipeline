#!/usr/bin/env python3
"""
Quick test script - processes 5 samples for validation.

This script provides a convenient entry point for testing the pipeline
with a small sample size before running on the full dataset.
"""
import subprocess
import sys

if __name__ == "__main__":
    print("="*60)
    print("TEST MODE: Processing 5 samples")
    print("="*60)
    sys.exit(subprocess.run([sys.executable, "main.py", "--test"]).returncode)

