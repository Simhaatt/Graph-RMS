"""Dispatch the frozen classical or class-count-free baseline runner."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("family", choices=("classical", "no-k"))
    args, remainder = parser.parse_known_args()
    target = (
        ROOT / "scripts/_canonical_baselines.py"
        if args.family == "classical"
        else ROOT / "scripts/_run_no_k_baselines.py"
    )
    subprocess.run([sys.executable, "-u", str(target), *remainder], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()

