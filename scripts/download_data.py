"""Acquire one or more public scenes through the frozen Graph-RMS loader."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from graphrms.data import load_scene

DATASETS = [
    "salinas_a", "indian_pines", "ksc", "pavia_university",
    "whu_hi_longkou", "whu_hi_honghu", "whu_hi_hanchuan", "botswana", "trento",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dataset", choices=DATASETS)
    group.add_argument("--all", action="store_true")
    group.add_argument("--list", action="store_true")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    args = parser.parse_args()
    if args.list:
        print("\n".join(DATASETS))
        return
    selected = DATASETS if args.all else [args.dataset]
    for dataset in selected:
        scene = load_scene(dataset, data_dir=args.data_dir)
        print(json.dumps({
            "dataset": dataset,
            "cube_shape": list(scene.cube.shape),
            "ground_truth_shape": list(scene.gt.shape),
            "labelled_pixels": int((scene.gt > 0).sum()),
            "data_dir": str(args.data_dir.resolve()),
        }))


if __name__ == "__main__":
    main()

