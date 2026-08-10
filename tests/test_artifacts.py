from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def count_rows(relative: str) -> int:
    with (ROOT / relative).open(newline="", encoding="utf-8-sig") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def test_frozen_registry_has_nine_scenes():
    registry = json.loads((ROOT / "results/provenance/frozen_results_registry.json").read_text())
    assert len(registry["datasets"]) == 9
    assert registry["holdout"] == "trento"


def test_evidence_row_counts():
    assert count_rows("results/main_tables/master_results_tidy.csv") == 63
    assert count_rows("results/main_tables/no_k_aggregate.csv") == 18
    assert count_rows("results/supporting_results/bootstrap_ci.csv") == 36
    assert count_rows("results/supporting_results/ablation_summary.csv") == 8
    assert count_rows("results/supporting_results/sensitivity_runs_raw.csv") == 51
    assert count_rows("results/supporting_results/profile_runs_raw.csv") == 27


def test_primary_partitions_present():
    registry = json.loads((ROOT / "results/provenance/frozen_results_registry.json").read_text())
    for dataset in registry["datasets"]:
        base = ROOT / "results/per_dataset" / dataset
        assert (base / "summary.json").exists()
        assert (base / "endpoint_curve.csv").exists()
        assert (base / "selected_labels.npy").exists()


def test_primary_partition_shapes_match_registry():
    registry = json.loads((ROOT / "results/provenance/frozen_results_registry.json").read_text())
    for dataset, info in registry["datasets"].items():
        labels = np.load(ROOT / "results/per_dataset" / dataset / "selected_labels.npy", mmap_mode="r")
        assert labels.shape == (info["height"], info["width"])
        assert np.unique(labels).size == info["n_clusters_full"]
