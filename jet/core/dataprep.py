#!/usr/bin/env python3
from jet.utils.files import PathFetcher

from bpekit import download_dataset, encode_dataset, train_tokenizer
import argparse
import subprocess
import sys
import shutil
import time

def is_slurm_available():
    """Check if SLURM is available."""
    return shutil.which('sbatch') is not None

def run_command(command):
    """Run a command directly and wait for it to complete."""
    result = subprocess.run(command, shell=True)
    if result.returncode != 0:
        print(f"Command failed: {command}", file=sys.stderr)
        sys.exit(1)

def main():
    pass

if __name__ == '__main__':
    main()
