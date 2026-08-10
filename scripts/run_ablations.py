"""Run the eight frozen Pavia University component-ablation configurations."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/ablation")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--save-labels", action="store_true")
    args = parser.parse_args()
    command = [
        sys.executable, "-u", str(ROOT / "scripts/_journal_response_studies.py"),
        "--study", "ablation", "--datasets", "pavia_university",
        "--data-dir", str(args.data_dir), "--output-dir", str(args.output_dir),
    ]
    if args.cpu:
        command.append("--cpu")
    if args.save_labels:
        command.append("--save-labels")
    subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()

