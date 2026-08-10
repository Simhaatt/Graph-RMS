"""Run the frozen primary Graph-RMS configuration on selected scenes."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASETS = [
    "salinas_a", "indian_pines", "ksc", "pavia_university",
    "whu_hi_longkou", "whu_hi_honghu", "whu_hi_hanchuan", "botswana", "trento",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--datasets", nargs="+", choices=DATASETS)
    group.add_argument("--all", action="store_true")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/primary")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--save-labels", action="store_true")
    args = parser.parse_args()
    command = [
        sys.executable, "-u", str(ROOT / "scripts/_journal_response_studies.py"),
        "--study", "profile", "--profile-repeats", "1",
        "--datasets", *(DATASETS if args.all else args.datasets),
        "--data-dir", str(args.data_dir), "--output-dir", str(args.output_dir),
    ]
    if args.cpu:
        command.append("--cpu")
    if args.save_labels:
        command.append("--save-labels")
    subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()

