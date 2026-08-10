"""Assemble fairness, resource, consolidation, and fragmentation evidence.

This script consumes frozen adaptive-v5 outputs. It performs no model selection
and does not alter any reported partition.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from graphrms.data import load_scene


REGISTRY_PATH = ROOT / "submission_package" / "data" / "frozen_results_registry.json"
METRICS_PATH = ROOT / "submission_package" / "data" / "dataset_method_metrics.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("journal_validation/existing_evidence"))
    parser.add_argument("--fragmentation-datasets", nargs="*",
                        default=["salinas_a", "indian_pines", "pavia_university"])
    return parser.parse_args()


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def selected_curve_row(summary: dict) -> dict:
    threshold = float(summary["label_free_selected"]["prototype_threshold"])
    return min(summary["curve"],
               key=lambda row: abs(float(row["prototype_threshold"]) - threshold))


def fragmentation_metrics(gt: np.ndarray, labels: np.ndarray) -> dict:
    gt = np.asarray(gt).reshape(-1)
    labels = np.asarray(labels).reshape(-1)
    mask = gt > 0
    yt = gt[mask].astype(np.int64)
    yp = labels[mask].astype(np.int64)
    pred_ids, pred_inv = np.unique(yp, return_inverse=True)
    true_ids, true_inv = np.unique(yt, return_inverse=True)
    contingency = np.zeros((pred_ids.size, true_ids.size), dtype=np.int64)
    np.add.at(contingency, (pred_inv, true_inv), 1)
    purity = float(contingency.max(axis=1).sum() / contingency.sum())

    fragments_90 = []
    dominant_fraction = []
    class_entropy = []
    for column in contingency.T:
        positive = column[column > 0]
        probabilities = positive / positive.sum()
        ordered = np.sort(probabilities)[::-1]
        fragments_90.append(int(np.searchsorted(np.cumsum(ordered), 0.90) + 1))
        dominant_fraction.append(float(ordered[0]))
        entropy = -float(np.sum(probabilities * np.log(probabilities + 1e-12)))
        class_entropy.append(
            entropy / max(float(np.log(positive.size)), 1e-12)
            if positive.size > 1 else 0.0
        )
    return {
        "predicted_clusters_on_labeled_pixels": int(pred_ids.size),
        "cluster_purity_many_to_one": purity,
        "mean_fragments_for_90pct_class_mass": float(np.mean(fragments_90)),
        "max_fragments_for_90pct_class_mass": int(np.max(fragments_90)),
        "mean_dominant_cluster_fraction_per_class": float(np.mean(dominant_fraction)),
        "mean_normalized_within_class_fragment_entropy": float(np.mean(class_entropy)),
    }


def fairness_rows() -> list[dict]:
    common = {
        "evaluation_mask": "reference labels used only for final scoring",
        "labelled_hyperparameter_tuning": "no",
    }
    return [
        {
            "method": "Graph-RMS",
            "family": "proposed",
            "receives_true_K": "no",
            "reported_repeats": "deterministic frozen partition",
            "implementation": "project implementation",
            "adaptation_or_shim": "none",
            "spectral_input": (
                "per-band z-score; PCA-20 only for graph neighbour search; "
                "diffusion uses all bands"
            ),
            **common,
        },
        {
            "method": "PCA-KMeans",
            "family": "classical",
            "receives_true_K": "yes",
            "reported_repeats": "5 seeds",
            "implementation": "scikit-learn",
            "adaptation_or_shim": "none",
            "spectral_input": "per-band StandardScaler followed by PCA-20",
            **common,
        },
        {
            "method": "MiniBatch-KMeans",
            "family": "classical",
            "receives_true_K": "yes",
            "reported_repeats": "5 seeds",
            "implementation": "scikit-learn",
            "adaptation_or_shim": "none",
            "spectral_input": "per-band StandardScaler followed by PCA-20",
            **common,
        },
        {
            "method": "FCM",
            "family": "classical",
            "receives_true_K": "yes",
            "reported_repeats": "5 seeds",
            "implementation": "project NumPy implementation",
            "adaptation_or_shim": "none",
            "spectral_input": "per-band StandardScaler followed by PCA-20",
            **common,
        },
        {
            "method": "SLIC-KMeans",
            "family": "classical",
            "receives_true_K": "yes",
            "reported_repeats": "5 seeds",
            "implementation": "scikit-image plus scikit-learn",
            "adaptation_or_shim": "none",
            "spectral_input": "per-band StandardScaler followed by PCA-20",
            **common,
        },
        {
            "method": "DLSS",
            "family": "modern HSI",
            "receives_true_K": "yes",
            "reported_repeats": "seed 0; seed-1 identity check on 3 scenes",
            "implementation": "public MATLAB/Octave source",
            "adaptation_or_shim": (
                "declared SLIC region adaptation replaces dense pixel-pair matrices"
            ),
            "spectral_input": "standardized PCA-20",
            **common,
        },
        {
            "method": "S2DL",
            "family": "modern HSI",
            "receives_true_K": "yes",
            "reported_repeats": "seed 0; seed-1 identity check on 3 scenes",
            "implementation": "public MATLAB/Octave source",
            "adaptation_or_shim": (
                "declared Octave graph, propagation, and superpixel compatibility shims"
            ),
            "spectral_input": "standardized PCA-20",
            **common,
        },
    ]


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    registry = json.loads(REGISTRY_PATH.read_text())
    metric_rows = read_csv(METRICS_PATH)
    graph_metrics = {
        row["dataset"]: row for row in metric_rows if row["method"] == "Graph-RMS"
    }

    runtime_rows = []
    overseg_rows = []
    summary_by_dataset: dict[str, dict] = {}
    for dataset, frozen in registry["datasets"].items():
        summary_path = ROOT / frozen["method_source"]
        summary = json.loads(summary_path.read_text())
        summary_by_dataset[dataset] = summary
        selected = selected_curve_row(summary)
        pixels = int(frozen["height"] * frozen["width"])
        runtime_rows.append({
            "dataset": dataset,
            "display_name": frozen["display_name"],
            "pixels": pixels,
            "bands": frozen["bands"],
            "device_record": (
                "GPU" if summary.get("peak_gpu_memory_mb") is not None else "CPU"
            ),
            "total_runtime_seconds": summary.get("total_runtime_seconds"),
            "peak_gpu_memory_mb": summary.get("peak_gpu_memory_mb"),
            **{f"stage_{key}": value for key, value
               in summary.get("stage_runtime_seconds", {}).items()},
            "source": frozen["method_source"],
        })
        labeled_clusters = int(selected.get(
            "n_pred_clusters_labeled", frozen["n_clusters_full"]
        ))
        oa = float(graph_metrics[dataset]["overall_accuracy_mean"])
        overseg_rows.append({
            "dataset": dataset,
            "display_name": frozen["display_name"],
            "true_classes": frozen["classes"],
            "fine_modes_full": summary["fine_modes"],
            "final_clusters_full": frozen["n_clusters_full"],
            "final_clusters_on_labeled_pixels": labeled_clusters,
            "full_cluster_to_class_ratio": (
                float(frozen["n_clusters_full"]) / float(frozen["classes"])
            ),
            "labeled_cluster_to_class_ratio": (
                float(labeled_clusters) / float(frozen["classes"])
            ),
            "fine_to_final_compression_ratio": (
                float(frozen["n_clusters_full"]) / float(summary["fine_modes"])
            ),
            "labeled_pixel_fraction": float(frozen["labeled_pixels"] / pixels),
            "OA_hungarian_one_to_one": oa,
            "NMI": float(graph_metrics[dataset]["nmi_mean"]),
            "ARI": float(graph_metrics[dataset]["ari_mean"]),
        })

    detailed = {}
    for dataset in args.fragmentation_datasets:
        frozen = registry["datasets"][dataset]
        labels_path = ROOT / frozen["method_source"]
        labels_path = labels_path.parent / f"{dataset}_selected_labels.npy"
        scene = load_scene(dataset, data_dir=args.data_dir)
        labels = np.load(labels_path)
        detail = fragmentation_metrics(scene.gt, labels)
        detailed[dataset] = detail
        row = next(row for row in overseg_rows if row["dataset"] == dataset)
        row.update(detail)
        row["purity_minus_hungarian_OA"] = (
            detail["cluster_purity_many_to_one"] - row["OA_hungarian_one_to_one"]
        )

    fairness = fairness_rows()
    write_csv(args.output_dir / "baseline_fairness.csv", fairness)
    write_csv(args.output_dir / "runtime_memory_existing.csv", runtime_rows)
    write_csv(args.output_dir / "oversegmentation_analysis.csv", overseg_rows)

    report_lines = [
        "# Journal-response evidence audit",
        "",
        "## Baseline fairness",
        "",
        "Graph-RMS never receives the reference class count. Every comparison "
        "method does. DLSS is a declared scalable region adaptation, and S2DL "
        "contains declared Octave compatibility shims; neither is described as "
        "an exact native MATLAB reproduction.",
        "",
        "## Runtime and memory",
        "",
        "Frozen runtime values exist for all nine scenes, but they were produced "
        "on mixed CPU/GPU environments. They support within-method scalability "
        "reporting, not a matched cross-method speed ranking. Peak GPU allocation "
        "exists only where the source summary records it; the new profiler should "
        "be used for synchronized CPU RSS and GPU allocation.",
        "",
        "## Over-segmentation interpretation",
        "",
        "Fine modes are an intentional intermediate representation, and prototype "
        "consolidation compresses them substantially. A final cluster count well "
        "above the semantic class count is nevertheless a limitation, not a "
        "feature: it indicates unresolved within-class spectral/spatial modes. "
        "NMI can remain high under such refinements, while one-to-one Hungarian "
        "OA and minority-sensitive BA can decrease. The many-to-one purity minus "
        "Hungarian-OA gap quantifies how much accuracy is hidden by class "
        "fragmentation rather than cross-class mixing.",
        "",
        "Detailed fragmentation measures were computed only for locally available "
        "ground-truth scenes; the same frozen analysis should be run in Colab for "
        "the remaining large scenes.",
    ]
    (args.output_dir / "evidence_audit.md").write_text("\n".join(report_lines))
    manifest = {
        "protocol": "frozen adaptive-v5 evidence-only analysis",
        "model_selection_performed": False,
        "baseline_fairness_rows": len(fairness),
        "runtime_rows": len(runtime_rows),
        "oversegmentation_rows": len(overseg_rows),
        "detailed_fragmentation_datasets": sorted(detailed),
        "outputs": [
            "baseline_fairness.csv",
            "runtime_memory_existing.csv",
            "oversegmentation_analysis.csv",
            "evidence_audit.md",
        ],
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
