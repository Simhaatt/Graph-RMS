"""Audit curated Graph-RMS evidence against manuscript-critical invariants."""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATASETS = [
    "salinas_a", "indian_pines", "ksc", "pavia_university",
    "whu_hi_longkou", "whu_hi_honghu", "whu_hi_hanchuan", "botswana", "trento",
]
EXPECTED_PRIMARY = {
    "salinas_a": (7, 0.8057, 0.7475, 0.8659, 0.7495),
    "indian_pines": (35, 0.4181, 0.4956, 0.5885, 0.3022),
    "ksc": (720, 0.4642, 0.3755, 0.6306, 0.3226),
    "pavia_university": (207, 0.6365, 0.5009, 0.6836, 0.5137),
    "whu_hi_longkou": (110, 0.8110, 0.5497, 0.8079, 0.8261),
    "whu_hi_honghu": (46, 0.7704, 0.6198, 0.7649, 0.8586),
    "whu_hi_hanchuan": (83, 0.5130, 0.3828, 0.6134, 0.6365),
    "botswana": (38, 0.4498, 0.4381, 0.7166, 0.3234),
    "trento": (68, 0.8533, 0.6774, 0.8495, 0.9447),
}


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def close4(actual: str | float, expected: float) -> bool:
    return math.isclose(round(float(actual), 4), expected, abs_tol=5e-5)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict-manuscript-assets", action="store_true")
    args = parser.parse_args()
    failures: list[str] = []
    warnings: list[str] = []

    required = [
        "README.md", "LICENSE", "CITATION.cff", ".zenodo.json",
        "configs/primary.yaml", "configs/automatic_selector.yaml",
        "results/provenance/frozen_results_registry.json",
        "results/main_tables/master_results_tidy.csv",
        "results/main_tables/no_k_aggregate.csv",
    ]
    for relative in required:
        if not (ROOT / relative).exists():
            failures.append(f"missing required artifact: {relative}")

    registry = json.loads((ROOT / "results/provenance/frozen_results_registry.json").read_text())
    if list(registry["datasets"]) != DATASETS:
        failures.append("frozen registry dataset order or membership differs from protocol")

    master = rows(ROOT / "results/main_tables/master_results_tidy.csv")
    if len(master) != 63:
        failures.append(f"master_results_tidy.csv has {len(master)} rows, expected 63")
    graph = {row["dataset"]: row for row in master if row["method"] == "Graph-RMS"}
    for dataset, expected in EXPECTED_PRIMARY.items():
        row = graph.get(dataset)
        if row is None:
            failures.append(f"missing Graph-RMS master row for {dataset}")
            continue
        observed = (
            int(float(row["n_clusters_full"])), row["overall_accuracy_mean"],
            row["balanced_accuracy_mean"], row["nmi_mean"], row["ari_mean"],
        )
        if observed[0] != expected[0] or any(
            not close4(value, target) for value, target in zip(observed[1:], expected[1:])
        ):
            failures.append(f"primary manuscript mismatch for {dataset}: {observed} vs {expected}")
        scene_dir = ROOT / "results/per_dataset" / dataset
        for filename in ("summary.json", "endpoint_curve.csv", "selected_labels.npy"):
            if not (scene_dir / filename).exists():
                failures.append(f"missing {dataset}/{filename}")

    expected_counts = {
        "results/supporting_results/bootstrap_ci.csv": 36,
        "results/supporting_results/ablation_summary.csv": 8,
        "results/supporting_results/sensitivity_runs_raw.csv": 51,
        "results/supporting_results/profile_runs_raw.csv": 27,
        "results/supporting_results/cross_device_repeatability.csv": 3,
        "results/supporting_results/pavia_adaptive_robustness.csv": 4,
        "results/automatic_selector/development_master_table.csv": 8,
    }
    for relative, expected in expected_counts.items():
        count = len(rows(ROOT / relative))
        if count != expected:
            failures.append(f"{relative} has {count} rows, expected {expected}")

    no_k = rows(ROOT / "results/main_tables/no_k_aggregate.csv")
    pairs = {(row["dataset"], row["method"]) for row in no_k}
    expected_pairs = {(dataset, method) for dataset in DATASETS for method in ("HDBSCAN", "Leiden")}
    if pairs != expected_pairs:
        failures.append("no-K aggregate does not contain all 18 dataset-method pairs")

    automatic = rows(ROOT / "results/automatic_selector/development_master_table.csv")
    statuses = [row["selector_status"] for row in automatic]
    if statuses.count("selected_size_aware") != 1 or statuses.count("fallback_conservative") != 7:
        failures.append(f"unexpected automatic-v2 branch counts: {statuses}")
    trento = json.loads((ROOT / "results/automatic_selector/trento_retrospective_audit.json").read_text())
    if not trento.get("not_a_new_untouched_holdout"):
        failures.append("automatic Trento audit lost retrospective-status flag")
    for key, expected in {
        "overall_accuracy": 0.6148, "balanced_accuracy": 0.5122,
        "nmi": 0.7656, "ari": 0.7164,
    }.items():
        if not close4(trento["evaluation_after_selection"][key], expected):
            failures.append(f"automatic Trento {key} mismatch")

    figures = [
        "fig2_representative_cluster_maps.png", "fig3_fragmentation.png",
        "fig4_sensitivity.png", "repeatability_and_perturbation_matched_style.png",
        "fig6_runtime_memory.png",
    ]
    for filename in figures:
        if not (ROOT / "figures" / filename).exists():
            failures.append(f"missing supplied manuscript figure: {filename}")
    if not (ROOT / "figures/graph.pdf").exists():
        message = "missing vector workflow figure: figures/graph.pdf"
        (failures if args.strict_manuscript_assets else warnings).append(message)
    if not (ROOT / "manuscript_support/references.bib").exists():
        message = "missing supplied bibliography: manuscript_support/references.bib"
        (failures if args.strict_manuscript_assets else warnings).append(message)

    manuscript = (ROOT / "manuscript_support/manuscript_supplied.tex").read_text(errors="replace")
    if manuscript.count(r"\section*{Data availability statement}") != 1:
        failures.append("manuscript must contain exactly one Data Availability section")
    if "zenodo.XXXXXXXX" in manuscript:
        failures.append("manuscript contains the prohibited fake Zenodo DOI")
    license_text = (ROOT / "LICENSE").read_text(encoding="utf-8")
    if not license_text.startswith("MIT License"):
        failures.append("approved MIT License text is missing")

    report = {
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "warnings": warnings,
        "datasets": len(DATASETS),
        "master_rows": len(master),
        "no_k_pairs": len(pairs),
    }
    print(json.dumps(report, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
