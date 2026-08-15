"""Run the recovered automatic-v2 Graph-RMS selector.

Candidate generation and target-scene selection do not load reference labels.
The separate ``evaluate`` command is the first step that accesses a reference
map.  ``verify-archive`` replays v2 from existing label-free endpoint caches
and checks the resulting partitions against the archived reported outputs.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import adjusted_rand_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import _run_automatic_selection as candidate_generator
from graphrms.automatic_v2 import (
    ALL_DATASETS,
    DEVELOPMENT_DATASETS,
    EXPECTED_RULE_SHA256,
    RETROSPECTIVE_DATASET,
    candidate_relationships,
    load_candidates,
    load_locked_rule,
    select_v2,
    select_from_cache,
)
from graphrms.data import load_scene
from graphrms.metrics import evaluate

DEFAULT_LOCK = ROOT / "results/automatic_selector/development_lock.json"
DEFAULT_ARCHIVE = ROOT / "results/automatic_selector"
DEFAULT_REGISTRY = ROOT / "results/provenance/frozen_results_registry.json"


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def _selected_metadata(chosen: dict, selector_audit: dict) -> dict:
    beta = float(chosen["size_exponent"])
    return {
        "checkpoint": int(chosen["checkpoint"]),
        "local_scale_s_T": float(chosen["local_scale_s_t"]),
        "gamma": float(chosen["gamma"]),
        "realized_fine_radius": float(chosen["fine_radius"]),
        "prototype_threshold": float(chosen["prototype_threshold"]),
        "size_exponent": beta,
        "regions_full_scene": int(chosen["n_clusters_full"]),
        "tau_neighbor_stability": float(chosen["tau_neighbor_stability"]),
        "checkpoint_stability": float(chosen["T_stability"]),
        "radius_stability": float(chosen["gamma_stability"]),
        "stable_support": float(
            max(chosen["T_stability"], chosen["gamma_stability"])
        ),
        "beta_agreement": float(chosen["beta_agreement"]),
        "stable_component_size": int(
            selector_audit["component_support_by_beta"][str(beta)]
        ),
        "stable_component_mean_stability": float(
            selector_audit["component_mean_stability_by_beta"][str(beta)]
        ),
        "largest_region_fraction": float(chosen["largest_cluster_fraction_full"]),
        "normalized_entropy": float(chosen["normalized_entropy_full"]),
        "compression": float(chosen["compression"]),
    }


def _candidate_protocol_lock(args: argparse.Namespace) -> Path | None:
    if args.dataset != RETROSPECTIVE_DATASET:
        return None
    if args.candidate_protocol_lock is None:
        raise RuntimeError(
            "Trento candidate generation requires the frozen v1 development "
            "protocol lock. Supply --candidate-protocol-lock, or point "
            "--candidate-root to complete existing endpoint caches."
        )
    return args.candidate_protocol_lock


def select_dataset(args: argparse.Namespace) -> None:
    lock_payload, rule = load_locked_rule(args.rule_lock)
    output_dir = args.output_root / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    selection_lock = output_dir / "v2_selection_lock.json"
    if selection_lock.exists():
        print(f"[skip] immutable automatic-v2 lock already exists: {selection_lock}")
        return

    cache_dir = args.candidate_root / args.dataset
    if not all(
        (cache_dir / f"checkpoint_T{checkpoint}_endpoints.npz").exists()
        for checkpoint in (25, 50, 100, 200)
    ):
        candidate_generator.select_dataset(
            args.dataset,
            args.data_dir,
            args.candidate_root,
            args.registry,
            _candidate_protocol_lock(args),
            args.cpu,
        )

    started = time.perf_counter()
    chosen, selector_audit = select_from_cache(cache_dir, rule)
    elapsed = time.perf_counter() - started
    common = {
        "protocol": "automatic-v2",
        "dataset": args.dataset,
        "v2_development_rule_sha256": EXPECTED_RULE_SHA256,
        "v2_development_lock_sha256": sha256(args.rule_lock),
        "candidate_source": "automatic-scale-v1 endpoint caches; v1 decision ignored",
        "candidate_generation_protocol_sha256": hashlib.sha256(
            json.dumps(candidate_generator.PROTOCOL, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "selection_uses_reference_labels": False,
        "global_rule_calibration_uses_development_labels": True,
        "trento_role": (
            "retrospective_transfer_not_second_holdout"
            if args.dataset == RETROSPECTIVE_DATASET
            else None
        ),
        "selector_audit": selector_audit,
        "v2_selection_runtime_seconds_from_existing_caches": elapsed,
        "runner_sha256": sha256(Path(__file__)),
        "development_lock_status": lock_payload["status"],
    }
    if chosen is None:
        payload = {**common, "status": "abstain"}
    else:
        registry = json.loads(args.registry.read_text(encoding="utf-8"))
        scene = registry["datasets"][args.dataset]
        labels = chosen["labels"].reshape(int(scene["height"]), int(scene["width"]))
        labels_path = output_dir / f"{args.dataset}_v2_selected_labels.npy"
        np.save(labels_path, labels.astype(np.int32, copy=False))
        selected = _selected_metadata(chosen, selector_audit)
        payload = {
            **common,
            "status": "selected",
            "selector_status": (
                "selected_size_aware"
                if selected["size_exponent"] == 0.5
                else "fallback_conservative"
            ),
            "selected": selected,
            "selected_labels": labels_path.name,
            "selected_labels_sha256": sha256(labels_path),
        }
    selection_lock.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


def evaluate_dataset(args: argparse.Namespace) -> None:
    output_dir = args.output_root / args.dataset
    lock_path = output_dir / "v2_selection_lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock["status"] != "selected":
        print(f"[skip evaluation] {args.dataset}: selector abstained")
        return
    labels_path = output_dir / lock["selected_labels"]
    if sha256(labels_path) != lock["selected_labels_sha256"]:
        raise RuntimeError("selected automatic-v2 partition hash mismatch")
    # This is intentionally the first target-scene step that loads labels.
    scene = load_scene(args.dataset, data_dir=args.data_dir)
    metrics = evaluate(scene.gt, np.load(labels_path))
    payload = {
        "dataset": args.dataset,
        "evaluation_after_selection_lock": True,
        "selection_lock_sha256": sha256(lock_path),
        **metrics,
    }
    target = output_dir / "v2_reference_evaluation.json"
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


def _close(left: float, right: float, tolerance: float = 1e-12) -> bool:
    return abs(float(left) - float(right)) <= tolerance


def verify_archive(args: argparse.Namespace) -> None:
    _, rule = load_locked_rule(args.rule_lock)
    datasets = args.datasets or ALL_DATASETS
    records = []
    for dataset in datasets:
        print(f"[automatic-v2 archive replay] {dataset}", flush=True)
        chosen, selector_audit = select_from_cache(args.candidate_root / dataset, rule)
        if chosen is None:
            raise RuntimeError(f"{dataset}: recovered automatic-v2 selector abstained")
        observed = _selected_metadata(chosen, selector_audit)
        if dataset in DEVELOPMENT_DATASETS:
            audit_path = args.archive_root / dataset / f"{dataset}_v2_audit.json"
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            expected_labels = np.load(
                args.archive_root / dataset / f"{dataset}_v2_selected_labels.npy"
            )
            partition_ari = float(
                adjusted_rand_score(expected_labels.reshape(-1), chosen["labels"])
            )
            expected = {
                key: audit[key]
                for key in (
                    "checkpoint",
                    "local_scale_s_T",
                    "gamma",
                    "realized_fine_radius",
                    "prototype_threshold",
                    "size_exponent",
                    "regions_full_scene",
                )
            }
        else:
            audit = json.loads(
                (args.archive_root / "trento_retrospective_audit.json").read_text(
                    encoding="utf-8"
                )
            )["selected"]
            partition_ari = None
            expected = {
                "checkpoint": audit["checkpoint_T"],
                "local_scale_s_T": audit["local_scale_s_T"],
                "gamma": audit["gamma"],
                "realized_fine_radius": audit["fine_radius"],
                "prototype_threshold": audit["prototype_threshold_tau"],
                "size_exponent": audit["size_exponent_beta"],
                "regions_full_scene": audit["full_scene_regions"],
            }

        mismatches = {
            key: {"observed": observed[key], "expected": value}
            for key, value in expected.items()
            if not _close(observed[key], value)
        }
        if mismatches:
            raise RuntimeError(f"{dataset}: automatic-v2 metadata mismatch: {mismatches}")
        if partition_ari is not None and partition_ari < 1.0 - 1e-12:
            raise RuntimeError(
                f"{dataset}: replayed partition ARI is {partition_ari}, expected 1.0"
            )
        records.append(
            {
                "dataset": dataset,
                "metadata_exact": True,
                "partition_ari_vs_archive": partition_ari,
                "selected": observed,
            }
        )
    payload = {
        "status": "PASS",
        "protocol": "automatic-v2",
        "development_rule_sha256": EXPECTED_RULE_SHA256,
        "candidate_cache_source": (
            "external label-free automatic-scale-v1 endpoint caches supplied "
            "during verification; cache files are regenerated, not redistributed"
        ),
        "datasets_verified": len(records),
        "recovered_source_provenance": [
            {
                "role": "eight-scene automatic-v2 development and calibration source",
                "sha256": "9e7ef5b4129ff5d5569ed546369eaf8bc42b4cf1e9282bd6c8d2e7e8683093c5",
            },
            {
                "role": "frozen-rule MUUFL automatic-v2 transfer source",
                "sha256": "8e0d7cc57ec3158ddb1b8a4359c5a027ad72b72e152047736a662e38246fdb67",
            },
        ],
        "records": records,
    }
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


def calibrate_development(args: argparse.Namespace) -> None:
    """Reproduce the declared label-using global v2 rule calibration."""
    lock, locked_rule = load_locked_rule(args.rule_lock)
    candidate_sets = {}
    relationships = {}
    reference = {}
    for dataset in DEVELOPMENT_DATASETS:
        print(f"[automatic-v2 calibration cache] {dataset}", flush=True)
        candidates = load_candidates(args.candidate_root / dataset)
        candidate_sets[dataset] = candidates
        relationships[dataset] = candidate_relationships(candidates)
        reference[dataset] = load_scene(dataset, data_dir=args.data_dir).gt.reshape(-1)

    evaluation_cache = {}
    records = []
    for stability, compression, beta_agreement in itertools.product(
        locked_rule["stability_grid"],
        locked_rule["compression_max_grid"],
        locked_rule["beta_agreement_grid"],
    ):
        trial_rule = {
            **locked_rule,
            "chosen_stability_min": stability,
            "chosen_compression_max": compression,
            "chosen_beta_agreement_min": beta_agreement,
        }
        metrics = []
        for dataset in DEVELOPMENT_DATASETS:
            chosen, _ = select_v2(
                candidate_sets[dataset], relationships[dataset], trial_rule
            )
            if chosen is None:
                continue
            key = (dataset, chosen["_candidate_index"])
            if key not in evaluation_cache:
                evaluation_cache[key] = evaluate(
                    reference[dataset], chosen["labels"]
                )
            metrics.append(evaluation_cache[key])
        row = {
            "stability_min": stability,
            "compression_max": compression,
            "beta_agreement_min": beta_agreement,
            "coverage": len(metrics),
            "mean_OA": float(np.mean([m["overall_accuracy"] for m in metrics])),
            "mean_BA": float(np.mean([m["balanced_accuracy"] for m in metrics])),
            "mean_NMI": float(np.mean([m["nmi"] for m in metrics])),
            "mean_ARI": float(np.mean([m["ari"] for m in metrics])),
            "structural_score": float(
                np.mean([(m["nmi"] + m["ari"]) / 2 for m in metrics])
            ),
        }
        records.append(row)

    ranked = sorted(
        records,
        key=lambda row: (
            row["coverage"],
            row["structural_score"],
            row["stability_min"],
            -row["compression_max"],
            row["beta_agreement_min"],
        ),
        reverse=True,
    )
    best = ranked[0]
    expected = {
        "stability_min": locked_rule["chosen_stability_min"],
        "compression_max": locked_rule["chosen_compression_max"],
        "beta_agreement_min": locked_rule["chosen_beta_agreement_min"],
        "coverage": lock["development_coverage"],
        "mean_OA": lock["development_metric_summary"]["OA"]["v2_mean"],
        "mean_BA": lock["development_metric_summary"]["BA"]["v2_mean"],
        "mean_NMI": lock["development_metric_summary"]["NMI"]["v2_mean"],
        "mean_ARI": lock["development_metric_summary"]["ARI"]["v2_mean"],
    }
    mismatches = {
        key: {"observed": best[key], "expected": value}
        for key, value in expected.items()
        if not _close(best[key], value)
    }
    if mismatches:
        raise RuntimeError(f"automatic-v2 calibration mismatch: {mismatches}")

    args.output_root.mkdir(parents=True, exist_ok=True)
    table = args.output_root / "development_rule_search.csv"
    with table.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    payload = {
        "status": "PASS",
        "protocol": "automatic-v2-global-calibration",
        "calibration_uses_development_labels": True,
        "target_scene_selection_uses_labels": False,
        "development_rule_sha256": EXPECTED_RULE_SHA256,
        "best": best,
        "expected": expected,
        "candidate_cache_source": "externally regenerated label-free endpoint caches",
        "rule_search_table": table.name,
    }
    report = args.output_root / "development_calibration_replay.json"
    report.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("select", "evaluate", "verify-archive", "calibrate-development"),
    )
    parser.add_argument("--dataset", choices=ALL_DATASETS)
    parser.add_argument("--datasets", nargs="*", choices=ALL_DATASETS)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs/automatic_v2")
    parser.add_argument("--candidate-root", type=Path)
    parser.add_argument("--candidate-protocol-lock", type=Path)
    parser.add_argument("--rule-lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--archive-root", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    if args.candidate_root is None:
        args.candidate_root = args.output_root / "candidate_generation_v1"
    if args.command in ("select", "evaluate") and args.dataset is None:
        parser.error(f"{args.command} requires --dataset")
    if args.command == "select":
        select_dataset(args)
    elif args.command == "evaluate":
        evaluate_dataset(args)
    elif args.command == "verify-archive":
        verify_archive(args)
    else:
        calibrate_development(args)


if __name__ == "__main__":
    main()
