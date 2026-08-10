"""Export manuscript-ready CSV subsets from the curated evidence tables."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATASET_ORDER = [
    "salinas_a", "indian_pines", "ksc", "pavia_university",
    "whu_hi_longkou", "whu_hi_honghu", "whu_hi_hanchuan", "botswana", "trento",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/derived/tables")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    master = pd.read_csv(ROOT / "results/main_tables/master_results_tidy.csv")
    master["dataset"] = pd.Categorical(master["dataset"], DATASET_ORDER, ordered=True)
    graph = master[master.method.eq("Graph-RMS")].sort_values("dataset")
    graph.to_csv(args.output_dir / "primary_results.csv", index=False)
    classical = master[master.method.isin([
        "Graph-RMS", "PCA-KMeans", "MiniBatch-KMeans", "FCM", "SLIC-KMeans"
    ])].sort_values(["dataset", "method"])
    classical.to_csv(args.output_dir / "classical_comparison.csv", index=False)
    no_k = pd.read_csv(ROOT / "results/main_tables/no_k_aggregate.csv")
    no_k.to_csv(args.output_dir / "no_k_baselines.csv", index=False)
    pd.read_csv(ROOT / "results/supporting_results/runtime_memory.csv").to_csv(
        args.output_dir / "runtime_memory.csv", index=False)
    pd.read_csv(ROOT / "results/automatic_selector/development_master_table.csv").to_csv(
        args.output_dir / "automatic_selector_development.csv", index=False)
    print(f"Exported tables to {args.output_dir}")


if __name__ == "__main__":
    main()

