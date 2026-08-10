"""Cluster-count controls and chance-corrected clustering metrics.

This analysis is deliberately separate from model development.  It never
changes a frozen Graph-RMS partition and never uses class identities for
selection.  Baselines are run with either:

* ``full_scene``: the number of clusters in the full Graph-RMS partition; or
* ``evaluated_support``: the number of Graph-RMS clusters intersecting the
  nonzero reference mask (a diagnostic control, clearly marked as such).

Every dataset/method/seed/budget run is checkpointed independently.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np
import psutil
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    homogeneity_completeness_v_measure,
    normalized_mutual_info_score,
)
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from graphrms.data import load_scene
from graphrms.metrics import evaluate


DEFAULT_REGISTRY = ROOT / "submission_package" / "data" / "frozen_results_registry.json"
METHODS = ("minibatch_kmeans", "pca_kmeans", "slic_kmeans", "fcm")
BUDGETS = ("full_scene", "evaluated_support")


class PeakRSS:
    def __init__(self, interval: float = 0.05):
        self.interval = interval
        self.peak = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self):
        process = psutil.Process(os.getpid())

        def sample() -> None:
            while not self._stop.is_set():
                self.peak = max(self.peak, process.memory_info().rss)
                self._stop.wait(self.interval)

        self._thread = threading.Thread(target=sample, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._stop.set()
        assert self._thread is not None
        self._thread.join()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "journal_validation" / "count_matched_controls",
    )
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument(
        "--methods", nargs="+", choices=METHODS, default=["minibatch_kmeans"]
    )
    parser.add_argument("--budgets", nargs="+", choices=BUDGETS, default=list(BUDGETS))
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--pca-components", type=int, default=20)
    parser.add_argument("--slic-region-size", type=int, default=100)
    parser.add_argument("--fcm-max-iter", type=int, default=150)
    parser.add_argument(
        "--fcm-max-memberships",
        type=int,
        default=80_000_000,
        help="Skip FCM when N*K exceeds this number of float memberships.",
    )
    return parser.parse_args()


def expanded_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    mask = y_true > 0
    yt, yp = y_true[mask], y_pred[mask]
    standard = evaluate(y_true, y_pred)
    homogeneity, completeness, v_measure = homogeneity_completeness_v_measure(yt, yp)
    return {
        "overall_accuracy": float(standard["overall_accuracy"]),
        "balanced_accuracy": float(standard["balanced_accuracy"]),
        "kappa": float(standard["kappa"]),
        "nmi": float(normalized_mutual_info_score(yt, yp, average_method="arithmetic")),
        "ami": float(adjusted_mutual_info_score(yt, yp, average_method="arithmetic")),
        "ari": float(adjusted_rand_score(yt, yp)),
        "homogeneity": float(homogeneity),
        "completeness": float(completeness),
        "v_measure": float(v_measure),
        "n_pred_clusters_labeled": int(np.unique(yp).size),
    }


def preprocess(spectra: np.ndarray, components: int) -> np.ndarray:
    x = StandardScaler(copy=True).fit_transform(spectra).astype(np.float32, copy=False)
    components = min(components, x.shape[1], x.shape[0] - 1)
    return PCA(
        n_components=components, svd_solver="randomized", random_state=0
    ).fit_transform(x).astype(np.float32, copy=False)


def slic_features(features: np.ndarray, h: int, w: int, target_size: int):
    from skimage.segmentation import slic

    image = features[:, : min(5, features.shape[1])].reshape(h, w, -1)
    lo = image.min(axis=(0, 1), keepdims=True)
    span = np.ptp(image, axis=(0, 1), keepdims=True)
    image = (image - lo) / np.maximum(span, 1e-8)
    requested = max(2, int(np.ceil(h * w / target_size)))
    regions = slic(
        image,
        n_segments=requested,
        compactness=0.1,
        channel_axis=-1,
        start_label=0,
        enforce_connectivity=True,
    ).reshape(-1)
    ids, inverse, counts = np.unique(regions, return_inverse=True, return_counts=True)
    sums = np.zeros((ids.size, features.shape[1]), dtype=np.float64)
    np.add.at(sums, inverse, features)
    means = (sums / counts[:, None]).astype(np.float32)
    return means, inverse, int(ids.size)


def fuzzy_cmeans(
    features: np.ndarray, n_clusters: int, seed: int, max_iter: int
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    memberships = rng.random((features.shape[0], n_clusters), dtype=np.float32)
    memberships /= memberships.sum(axis=1, keepdims=True)
    x2 = np.sum(features * features, axis=1, keepdims=True)
    for _ in range(max_iter):
        squared = memberships * memberships
        centers = (squared.T @ features) / np.maximum(
            squared.sum(axis=0)[:, None], 1e-12
        )
        dist2 = np.maximum(
            x2
            + np.sum(centers * centers, axis=1)[None, :]
            - 2 * features @ centers.T,
            1e-12,
        )
        updated = 1.0 / dist2
        updated /= updated.sum(axis=1, keepdims=True)
        delta = float(np.max(np.abs(updated - memberships)))
        memberships = updated.astype(np.float32, copy=False)
        if delta < 1e-4:
            break
    return memberships.argmax(axis=1).astype(np.int32)


def run_method(
    method: str,
    features: np.ndarray,
    n_clusters: int,
    seed: int,
    h: int,
    w: int,
    slic_cache,
    args: argparse.Namespace,
) -> np.ndarray:
    if method == "minibatch_kmeans":
        return MiniBatchKMeans(
            n_clusters=n_clusters,
            n_init=20,
            batch_size=4096,
            max_iter=300,
            random_state=seed,
        ).fit_predict(features)
    if method == "pca_kmeans":
        return KMeans(
            n_clusters=n_clusters, n_init=20, random_state=seed
        ).fit_predict(features)
    if method == "slic_kmeans":
        region_features, pixel_to_region, n_regions = slic_cache
        if n_clusters > n_regions:
            raise RuntimeError(
                f"requested K={n_clusters} exceeds {n_regions} SLIC regions"
            )
        region_labels = KMeans(
            n_clusters=n_clusters, n_init=20, random_state=seed
        ).fit_predict(region_features)
        return region_labels[pixel_to_region]
    if method == "fcm":
        if features.shape[0] * n_clusters > args.fcm_max_memberships:
            raise RuntimeError(
                f"FCM skipped by memory guard: N*K={features.shape[0] * n_clusters:,}"
            )
        return fuzzy_cmeans(features, n_clusters, seed, args.fcm_max_iter)
    raise ValueError(method)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    registry = json.loads(args.registry.read_text())
    datasets = args.datasets or list(registry["datasets"])
    unknown = sorted(set(datasets) - set(registry["datasets"]))
    if unknown:
        raise ValueError(f"datasets absent from registry: {unknown}")

    graph_rows: list[dict] = []
    run_records: list[dict] = []

    for dataset in datasets:
        info = registry["datasets"][dataset]
        scene = load_scene(dataset, data_dir=args.data_dir)
        h, w, bands = scene.cube.shape
        spectra = scene.cube.reshape(-1, bands).astype(np.float32, copy=False)
        gt = np.asarray(scene.gt).reshape(-1)

        summary_path = ROOT / info["method_source"]
        labels_path = summary_path.parent / f"{dataset}_selected_labels.npy"
        graph_labels = np.load(labels_path).reshape(-1)
        if graph_labels.size != gt.size:
            raise RuntimeError(
                f"{dataset}: Graph labels have {graph_labels.size} entries, GT has {gt.size}"
            )
        graph_metric = expanded_metrics(gt, graph_labels)
        full_k = int(np.unique(graph_labels).size)
        support_k = int(graph_metric["n_pred_clusters_labeled"])
        graph_rows.append(
            {
                "dataset": dataset,
                "display_name": info["display_name"],
                "method": "Graph-RMS",
                "budget": "label_free_partition",
                "requested_clusters": "",
                "full_scene_clusters": full_k,
                **graph_metric,
            }
        )
        print(
            f"[graph] {dataset}: K_full={full_k}, K_eval={support_k}, "
            f"NMI={graph_metric['nmi']:.4f}, AMI={graph_metric['ami']:.4f}, "
            f"H/C/V={graph_metric['homogeneity']:.4f}/"
            f"{graph_metric['completeness']:.4f}/{graph_metric['v_measure']:.4f}",
            flush=True,
        )

        if not args.methods:
            continue
        features = preprocess(spectra, args.pca_components)
        slic_cache = (
            slic_features(features, h, w, args.slic_region_size)
            if "slic_kmeans" in args.methods
            else None
        )
        requested_by_budget = {
            "full_scene": full_k,
            "evaluated_support": support_k,
        }

        for budget in args.budgets:
            requested_k = requested_by_budget[budget]
            for method in args.methods:
                for seed in args.seeds:
                    run_dir = args.output_dir / "runs" / dataset / budget / method
                    run_dir.mkdir(parents=True, exist_ok=True)
                    checkpoint = run_dir / f"seed_{seed}.json"
                    if checkpoint.exists():
                        record = json.loads(checkpoint.read_text())
                        run_records.append(record)
                        print(f"[skip] {dataset} {budget} {method} seed={seed}", flush=True)
                        continue
                    record = {
                        "dataset": dataset,
                        "display_name": info["display_name"],
                        "method": method,
                        "seed": seed,
                        "budget": budget,
                        "budget_interpretation": (
                            "primary full-scene partition-complexity control"
                            if budget == "full_scene"
                            else "diagnostic evaluated-support control; uses mask geometry, not class identities"
                        ),
                        "requested_clusters": requested_k,
                        "graph_full_scene_clusters": full_k,
                        "graph_evaluated_support_clusters": support_k,
                        "status": "complete",
                    }
                    start = time.perf_counter()
                    try:
                        with PeakRSS() as memory:
                            labels = run_method(
                                method,
                                features,
                                requested_k,
                                seed,
                                h,
                                w,
                                slic_cache,
                                args,
                            )
                            metrics = expanded_metrics(gt, labels)
                        record.update(metrics)
                        record["full_scene_clusters"] = int(np.unique(labels).size)
                        record["runtime_seconds"] = time.perf_counter() - start
                        record["peak_rss_mb"] = memory.peak / 1e6
                    except Exception as exc:
                        record["status"] = "skipped_or_failed"
                        record["reason"] = f"{type(exc).__name__}: {exc}"
                        record["runtime_seconds"] = time.perf_counter() - start
                    checkpoint.write_text(json.dumps(record, indent=2))
                    run_records.append(record)
                    print(json.dumps(record), flush=True)

    write_csv(args.output_dir / "graph_rms_information_metrics.csv", graph_rows)
    write_csv(args.output_dir / "count_matched_per_seed.csv", run_records)

    complete = [row for row in run_records if row["status"] == "complete"]
    aggregate_rows: list[dict] = []
    group_keys = sorted(
        {(row["dataset"], row["budget"], row["method"]) for row in complete}
    )
    metric_keys = (
        "overall_accuracy",
        "balanced_accuracy",
        "kappa",
        "nmi",
        "ami",
        "ari",
        "homogeneity",
        "completeness",
        "v_measure",
        "n_pred_clusters_labeled",
        "runtime_seconds",
        "peak_rss_mb",
    )
    for dataset, budget, method in group_keys:
        group = [
            row
            for row in complete
            if (row["dataset"], row["budget"], row["method"])
            == (dataset, budget, method)
        ]
        item = {
            "dataset": dataset,
            "display_name": group[0]["display_name"],
            "budget": budget,
            "method": method,
            "requested_clusters": group[0]["requested_clusters"],
            "n_repeats": len(group),
        }
        for key in metric_keys:
            values = np.asarray([row[key] for row in group], dtype=float)
            item[f"{key}_mean"] = float(values.mean())
            item[f"{key}_std"] = (
                float(values.std(ddof=1)) if values.size > 1 else 0.0
            )
        aggregate_rows.append(item)
    write_csv(args.output_dir / "count_matched_aggregate.csv", aggregate_rows)

    manifest = {
        "status": "complete",
        "registry": str(args.registry),
        "datasets": datasets,
        "methods": args.methods,
        "budgets": args.budgets,
        "seeds": args.seeds,
        "graph_rows": len(graph_rows),
        "baseline_runs_complete": len(complete),
        "baseline_runs_skipped_or_failed": len(run_records) - len(complete),
        "protocol": {
            "cluster_scope": "all scene pixels",
            "evaluation_scope": "reference labels > 0",
            "preprocessing": "per-band StandardScaler then randomized PCA-20",
            "labelled_tuning": False,
            "full_scene_budget": "primary count-matched control",
            "evaluated_support_budget": (
                "diagnostic only; mask geometry is used to define the requested count"
            ),
        },
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
