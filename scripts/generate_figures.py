"""Assemble the preserved manuscript figure assets without redesigning them."""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIGURES = [
    "graph.png",
    "fig2_representative_cluster_maps.png",
    "fig3_fragmentation.png",
    "fig4_sensitivity.png",
    "repeatability_and_perturbation_matched_style.png",
    "fig6_runtime_memory.png",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/derived/figures")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name in FIGURES:
        source = ROOT / "figures" / name
        if not source.exists():
            raise FileNotFoundError(source)
        shutil.copy2(source, args.output_dir / name)
    print(f"Assembled {len(FIGURES)} preserved figures in {args.output_dir}")


if __name__ == "__main__":
    main()
