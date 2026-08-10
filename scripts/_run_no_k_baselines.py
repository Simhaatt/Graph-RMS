"""Resumable no-K baselines for the Graph-RMS journal comparison.

Methods
-------
Leiden
    Runs on the exact frozen Graph-RMS weighted spectral-spatial MNN graph.
    Its resolution is selected without labels from a fixed grid using
    cross-seed and adjacent-resolution partition stability, together with the
    same largest-cluster and normalized-entropy degeneration gates used by the
    Graph-RMS selector.

HDBSCAN
    ``sklearn.cluster.HDBSCAN`` on the shared StandardScaler + PCA-20
    representation, with the library defaults unless explicitly overridden.
    It receives neither K nor labels. Noise remains label -1 and its fraction
    is reported.

Reference labels are accessed only after selection/fitting is complete. Every
expensive unit is checkpointed independently so rerunning resumes safely.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import sys
import threading
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import psutil
from sklearn.cluster import HDBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from graphrms.affinity import compute_gaussian_affinities
from graphrms.data import load_scene
from graphrms.graph import build_graph
from graphrms.metrics import evaluate


DATASETS = (
    "salinas_a",
    "indian_pines",
    "ksc",
    "pavia_university",
    "whu_hi_longkou",
    "whu_hi_honghu",
    "whu_hi_hanchuan",
    "botswana",
    "trento",
)
METHODS = ("leiden", "hdbscan")
DEFAULT_RESOLUTIONS = (0.03, 0.06, 0.10, 0.18, 0.32, 0.56, 1.0, 1.8, 3.2, 5.6)
METRIC_KEYS = (
    "overall_accuracy",
    "balanced_accuracy",
    "kappa",
    "nmi",
    "ami",
    "ari",
    "homogeneity",
    "completeness",
    "v_measure",
    "n_pred_clusters",
    "tiny_cluster_pixel_fraction",
)


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
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "no_k_baseline_results"
    )
    parser.add_argument(
        "--resolutions",
        type=float,
        nargs="+",
        default=list(DEFAULT_RESOLUTIONS),
    )
    parser.add_argument("--selector-seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--evaluation-seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--pca-components", type=int, default=20)
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--window-radius", type=int, default=7)
    parser.add_argument("--min-degree", type=int, default=4)
    parser.add_argument("--largest-cluster-cap", type=float, default=0.40)
    parser.add_argument("--entropy-floor", type=float, default=0.35)
    parser.add_argument("--hdbscan-min-cluster-size", type=int, default=5)
    parser.add_argument("--hdbscan-min-samples", type=int, default=None)
    parser.add_argument("--hdbscan-leaf-size", type=int, default=40)
    parser.add_argument("--n-jobs", type=int, default=-1)
    return parser.parse_args()


def band_zscore(cube: np.ndarray) -> np.ndarray:
    flat = cube.reshape(-1, cube.shape[-1])
    mean = flat.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = flat.std(axis=0, dtype=np.float64).astype(np.float32)
    std[std < 1e-8] = 1.0
    return ((cube - mean) / std).astype(np.float32)


def pca20(cube: np.ndarray, components: int) -> np.ndarray:
    flat = cube.reshape(-1, cube.shape[-1]).astype(np.float32, copy=False)
    standardized = StandardScaler(copy=True).fit_transform(flat).astype(
        np.float32, copy=False
    )
    n_components = min(components, standardized.shape[1], standardized.shape[0] - 1)
    return PCA(
        n_components=n_components,
        svd_solver="randomized",
        random_state=0,
    ).fit_transform(standardized).astype(np.float32, copy=False)


def partition_diagnostics(labels: np.ndarray) -> dict:
    labels = np.asarray(labels).reshape(-1)
    _, counts = np.unique(labels, return_counts=True)
    fractions = counts.astype(np.float64) / labels.size
    n_clusters = int(counts.size)
    entropy = -float(np.sum(fractions * np.log(fractions + 1e-15)))
    normalized_entropy = (
        entropy / float(np.log(n_clusters)) if n_clusters > 1 else 0.0
    )
    tiny_mass = float(fractions[counts < 0.005 * labels.size].sum())
    return {
        "n_clusters_full": n_clusters,
        "largest_cluster_fraction_full": float(fractions.max()),
        "normalized_cluster_entropy_full": normalized_entropy,
        "microcluster_pixel_fraction_full": tiny_mass,
    }


def json_dump(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def load_or_build_graph(
    dataset: str,
    cube: np.ndarray,
    output_dir: Path,
    args: argparse.Namespace,
):
    cache = output_dir / "graph_cache" / f"{dataset}_weighted_graph.npz"
    metadata_path = cache.with_suffix(".json")
    if cache.exists() and metadata_path.exists():
        print(f"[graph] {dataset}: loading cached weighted graph", flush=True)
        arrays = np.load(cache)
        metadata = json.loads(metadata_path.read_text())
        return (
            arrays["edge_i"],
            arrays["edge_j"],
            arrays["weight"],
            metadata,
        )

    print(f"[graph] {dataset}: building exact weighted graph", flush=True)
    started = time.perf_counter()
    graph_cube = band_zscore(cube)
    spectra = graph_cube.reshape(-1, graph_cube.shape[-1])
    graph = build_graph(
        graph_cube,
        pca_components=args.pca_components,
        k=args.k,
        window_radius=args.window_radius,
        mutual=True,
        min_degree=args.min_degree,
    )
    weights, bandwidths = compute_gaussian_affinities(
        spectra,
        graph.edge_i,
        graph.edge_j,
        graph.spatial_dist,
        seed=0,
    )
    # The graph stores both directions. Community detection receives one
    # undirected copy of each edge.
    keep = graph.edge_i < graph.edge_j
    edge_i = graph.edge_i[keep].astype(np.int32, copy=False)
    edge_j = graph.edge_j[keep].astype(np.int32, copy=False)
    weights = weights[keep].astype(np.float32, copy=False)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, edge_i=edge_i, edge_j=edge_j, weight=weights)
    metadata = {
        "dataset": dataset,
        "n_nodes": int(graph.n_nodes),
        "undirected_edges": int(edge_i.size),
        "pca_components_neighbor_search": args.pca_components,
        "k": args.k,
        "window_radius": args.window_radius,
        "mutual": True,
        "min_degree": args.min_degree,
        "sigma_spectral": float(bandwidths.sigma_spectral),
        "sigma_spatial": float(bandwidths.sigma_spatial),
        "build_seconds": time.perf_counter() - started,
    }
    json_dump(metadata_path, metadata)
    print(
        f"[graph] {dataset}: {metadata['n_nodes']:,} nodes, "
        f"{metadata['undirected_edges']:,} edges in "
        f"{metadata['build_seconds']:.1f} s",
        flush=True,
    )
    return edge_i, edge_j, weights, metadata


def make_igraph(n_nodes: int, edge_i: np.ndarray, edge_j: np.ndarray, weights: np.ndarray):
    import igraph as ig

    edges = np.column_stack((edge_i, edge_j))
    graph = ig.Graph(n=n_nodes, edges=edges, directed=False)
    graph.es["weight"] = weights.astype(float, copy=False)
    return graph


def leiden_partition(graph, resolution: float, seed: int) -> np.ndarray:
    import leidenalg

    partition = leidenalg.find_partition(
        graph,
        leidenalg.RBConfigurationVertexPartition,
        weights="weight",
        resolution_parameter=float(resolution),
        # Two iterations preserve the standard Leiden refinement pass and
        # were fixed before the nine-scene run.  On the diagnostic scene this
        # was 4.4--8.0x faster than convergence, without label-based tuning.
        n_iterations=2,
        seed=int(seed),
    )
    return np.asarray(partition.membership, dtype=np.int32)


def mean_pairwise_ari(partitions: list[np.ndarray]) -> float:
    if len(partitions) < 2:
        return 1.0
    return float(
        np.mean(
            [
                adjusted_rand_score(partitions[i], partitions[j])
                for i, j in combinations(range(len(partitions)), 2)
            ]
        )
    )


def mean_cross_resolution_ari(
    left: dict[int, np.ndarray], right: dict[int, np.ndarray], seeds: list[int]
) -> float:
    return float(
        np.mean(
            [adjusted_rand_score(left[seed], right[seed]) for seed in seeds]
        )
    )


def select_resolution_label_free(
    rows: list[dict],
    partitions: dict[float, dict[int, np.ndarray]],
    resolutions: list[float],
    seeds: list[int],
    largest_cap: float,
    entropy_floor: float,
) -> tuple[dict, dict]:
    by_resolution = {float(row["resolution"]): row for row in rows}
    for index, resolution in enumerate(resolutions):
        agreements = []
        if index > 0:
            agreements.append(
                mean_cross_resolution_ari(
                    partitions[resolutions[index - 1]],
                    partitions[resolution],
                    seeds,
                )
            )
        if index + 1 < len(resolutions):
            agreements.append(
                mean_cross_resolution_ari(
                    partitions[resolution],
                    partitions[resolutions[index + 1]],
                    seeds,
                )
            )
        row = by_resolution[resolution]
        row["adjacent_resolution_ari_label_free"] = (
            float(np.mean(agreements)) if agreements else 1.0
        )
        row["stability_score_label_free"] = min(
            float(row["cross_seed_ari_label_free"]),
            float(row["adjacent_resolution_ari_label_free"]),
        )
        row["interior_grid_point"] = bool(0 < index < len(resolutions) - 1)
        row["eligible_label_free"] = bool(
            row["interior_grid_point"]
            and
            row["n_clusters_full"] >= 2
            and row["largest_cluster_fraction_full"] <= largest_cap
            and row["normalized_cluster_entropy_full"] >= entropy_floor
        )

    candidates = [row for row in rows if row["eligible_label_free"]]
    if not candidates:
        # Conservative, label-free fallback: exclude only a one-cluster collapse
        # and minimize gate violations before considering stability.
        nontrivial = [row for row in rows if row["n_clusters_full"] >= 2]
        if not nontrivial:
            raise RuntimeError("all Leiden resolutions collapsed to one cluster")
        chosen = min(
            nontrivial,
            key=lambda row: (
                max(0.0, row["largest_cluster_fraction_full"] - largest_cap)
                + max(0.0, entropy_floor - row["normalized_cluster_entropy_full"]),
                -row["stability_score_label_free"],
                row["resolution"],
            ),
        )
        status = "fallback_minimum_degeneracy_violation"
    else:
        chosen = max(
            candidates,
            key=lambda row: (
                row["stability_score_label_free"],
                row["adjacent_resolution_ari_label_free"],
                row["cross_seed_ari_label_free"],
                row["normalized_cluster_entropy_full"],
                -row["resolution"],
            ),
        )
        status = "selected_stable_non_degenerate_resolution"

    return chosen, {
        "status": status,
        "selected_resolution": chosen["resolution"],
        "selection_uses_ground_truth": False,
        "selection_fields": [
            "cross_seed_ari_label_free",
            "adjacent_resolution_ari_label_free",
            "largest_cluster_fraction_full",
            "normalized_cluster_entropy_full",
            "n_clusters_full",
            "interior_grid_point",
        ],
        "stability_score": "minimum of cross-seed and adjacent-resolution ARI",
        "largest_cluster_fraction_cap": largest_cap,
        "normalized_entropy_floor": entropy_floor,
        "tie_break": (
            "higher minimum stability, adjacent stability, seed stability, "
            "entropy; then lower resolution"
        ),
        "resolution_grid_fixed_before_evaluation": resolutions,
        "boundary_rule": "endpoints are ineligible because they lack two-sided stability evidence",
        "selector_seeds": seeds,
        "ground_truth_loaded_after_selection": True,
    }


def add_evaluation(
    record: dict, gt: np.ndarray, labels: np.ndarray, noise_label: int | None = None
) -> dict:
    metrics = evaluate(gt.reshape(-1), labels.reshape(-1))
    record.update({key: metrics[key] for key in METRIC_KEYS})
    record["n_clusters_full"] = int(np.unique(labels).size)
    record["n_clusters_labeled"] = int(metrics["n_pred_clusters"])
    if noise_label is not None:
        record["noise_fraction_full"] = float(np.mean(labels == noise_label))
        mask = gt.reshape(-1) > 0
        record["noise_fraction_labeled"] = float(
            np.mean(labels.reshape(-1)[mask] == noise_label)
        )
        record["n_clusters_full_excluding_noise"] = int(
            np.unique(labels[labels != noise_label]).size
        )
        labelled_predictions = labels.reshape(-1)[mask]
        record["n_clusters_labeled_excluding_noise"] = int(
            np.unique(labelled_predictions[labelled_predictions != noise_label]).size
        )
    return record


def run_leiden(
    dataset: str,
    cube: np.ndarray,
    gt: np.ndarray,
    dataset_dir: Path,
    args: argparse.Namespace,
) -> list[dict]:
    edge_i, edge_j, weights, graph_metadata = load_or_build_graph(
        dataset, cube, args.output_dir, args
    )
    graph = make_igraph(cube.shape[0] * cube.shape[1], edge_i, edge_j, weights)
    resolutions = sorted({float(value) for value in args.resolutions})
    selector_seeds = sorted(set(args.selector_seeds))
    partitions: dict[float, dict[int, np.ndarray]] = {}
    surface_rows = []

    for resolution in resolutions:
        partitions[resolution] = {}
        runtimes = []
        for seed in selector_seeds:
            label_path = (
                dataset_dir
                / "leiden"
                / "surface_labels"
                / f"resolution_{resolution:g}_seed_{seed}.npy"
            )
            timing_path = label_path.with_suffix(".json")
            if label_path.exists() and timing_path.exists():
                print(
                    f"[leiden] {dataset}: resolution={resolution:g}, "
                    f"seed={seed} [cached]",
                    flush=True,
                )
                labels = np.load(label_path)
                timing = json.loads(timing_path.read_text())
            else:
                print(
                    f"[leiden] {dataset}: resolution={resolution:g}, "
                    f"seed={seed}",
                    flush=True,
                )
                started = time.perf_counter()
                with PeakRSS() as memory:
                    labels = leiden_partition(graph, resolution, seed)
                timing = {
                    "runtime_seconds": time.perf_counter() - started,
                    "peak_rss_mb": memory.peak / 1e6,
                }
                label_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(label_path, labels)
                json_dump(timing_path, timing)
                print(
                    f"[leiden] completed in {timing['runtime_seconds']:.1f} s; "
                    f"clusters={np.unique(labels).size}",
                    flush=True,
                )
            partitions[resolution][seed] = labels
            runtimes.append(timing)
        diagnostics = partition_diagnostics(partitions[resolution][selector_seeds[0]])
        surface_rows.append(
            {
                "dataset": dataset,
                "resolution": resolution,
                **diagnostics,
                "cross_seed_ari_label_free": mean_pairwise_ari(
                    [partitions[resolution][seed] for seed in selector_seeds]
                ),
                "runtime_seconds_mean": float(
                    np.mean([row["runtime_seconds"] for row in runtimes])
                ),
                "peak_rss_mb_max": float(
                    np.max([row["peak_rss_mb"] for row in runtimes])
                ),
            }
        )

    selected, audit = select_resolution_label_free(
        surface_rows,
        partitions,
        resolutions,
        selector_seeds,
        args.largest_cluster_cap,
        args.entropy_floor,
    )
    selection_path = dataset_dir / "leiden" / "selection.json"
    json_dump(
        selection_path,
        {
            "dataset": dataset,
            "method": "Leiden weighted RB-configuration partition",
            "graph": graph_metadata,
            "selected": selected,
            "audit": audit,
            "surface": surface_rows,
        },
    )

    # Only now are reference labels used.
    selected_resolution = float(selected["resolution"])
    print(
        f"[leiden] {dataset}: selected resolution={selected_resolution:g} "
        f"without labels ({audit['status']})",
        flush=True,
    )
    records = []
    for seed in sorted(set(args.evaluation_seeds)):
        if seed in partitions[selected_resolution]:
            labels = partitions[selected_resolution][seed]
            timing = json.loads(
                (
                    dataset_dir
                    / "leiden"
                    / "surface_labels"
                    / f"resolution_{selected_resolution:g}_seed_{seed}.json"
                ).read_text()
            )
        else:
            label_path = (
                dataset_dir
                / "leiden"
                / "selected_labels"
                / f"resolution_{selected_resolution:g}_seed_{seed}.npy"
            )
            timing_path = label_path.with_suffix(".json")
            if label_path.exists() and timing_path.exists():
                labels = np.load(label_path)
                timing = json.loads(timing_path.read_text())
            else:
                started = time.perf_counter()
                with PeakRSS() as memory:
                    labels = leiden_partition(graph, selected_resolution, seed)
                timing = {
                    "runtime_seconds": time.perf_counter() - started,
                    "peak_rss_mb": memory.peak / 1e6,
                }
                label_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(label_path, labels)
                json_dump(timing_path, timing)
        record = {
            "dataset": dataset,
            "method": "Leiden",
            "seed": seed,
            "uses_true_k": False,
            "uses_labels_for_selection": False,
            "selected_resolution": selected_resolution,
            **timing,
        }
        records.append(add_evaluation(record, gt, labels))
    return records


def run_hdbscan(
    dataset: str,
    cube: np.ndarray,
    gt: np.ndarray,
    dataset_dir: Path,
    args: argparse.Namespace,
) -> list[dict]:
    result_dir = dataset_dir / "hdbscan"
    label_path = result_dir / "labels.npy"
    record_path = result_dir / "record.json"
    if label_path.exists() and record_path.exists():
        labels = np.load(label_path)
        record = json.loads(record_path.read_text())
        if all(key in record for key in METRIC_KEYS):
            print(f"[hdbscan] {dataset}: loading cached result", flush=True)
            return [record]
    print(
        f"[hdbscan] {dataset}: fitting sklearn defaults on "
        f"{cube.shape[0] * cube.shape[1]:,} pixels; this stage can be quiet "
        "for a long time",
        flush=True,
    )
    started = time.perf_counter()
    with PeakRSS() as memory:
        features = pca20(cube, args.pca_components)
        model = HDBSCAN(
            min_cluster_size=args.hdbscan_min_cluster_size,
            min_samples=args.hdbscan_min_samples,
            cluster_selection_epsilon=0.0,
            max_cluster_size=None,
            metric="euclidean",
            alpha=1.0,
            algorithm="auto",
            leaf_size=args.hdbscan_leaf_size,
            n_jobs=args.n_jobs,
            cluster_selection_method="eom",
            allow_single_cluster=False,
            store_centers=None,
            copy=True,
        )
        labels = model.fit_predict(features).astype(np.int32)
    record = {
        "dataset": dataset,
        "method": "HDBSCAN",
        "seed": "",
        "uses_true_k": False,
        "uses_labels_for_selection": False,
        "preprocessing": "per-band StandardScaler then randomized PCA-20",
        "implementation": "sklearn.cluster.HDBSCAN",
        "min_cluster_size": args.hdbscan_min_cluster_size,
        "min_samples": args.hdbscan_min_samples,
        "cluster_selection_method": "eom",
        "runtime_seconds": time.perf_counter() - started,
        "peak_rss_mb": memory.peak / 1e6,
    }
    add_evaluation(record, gt, labels, noise_label=-1)
    result_dir.mkdir(parents=True, exist_ok=True)
    np.save(label_path, labels)
    json_dump(record_path, record)
    return [record]


def aggregate(records: list[dict]) -> list[dict]:
    rows = []
    keys = sorted({(row["dataset"], row["method"]) for row in records})
    for dataset, method in keys:
        group = [
            row
            for row in records
            if row["dataset"] == dataset and row["method"] == method
        ]
        item = {
            "dataset": dataset,
            "method": method,
            "uses_true_k": False,
            "uses_labels_for_selection": False,
            "n_repeats": len(group),
        }
        for key in (
            *METRIC_KEYS,
            "n_clusters_full",
            "n_clusters_labeled",
            "n_clusters_full_excluding_noise",
            "n_clusters_labeled_excluding_noise",
            "runtime_seconds",
            "peak_rss_mb",
            "noise_fraction_full",
            "noise_fraction_labeled",
        ):
            values = [row.get(key) for row in group if row.get(key) not in (None, "")]
            if not values:
                continue
            array = np.asarray(values, dtype=float)
            item[f"{key}_mean"] = float(array.mean())
            item[f"{key}_std"] = (
                float(array.std(ddof=1)) if array.size > 1 else 0.0
            )
        rows.append(item)
    return rows


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


def collect_existing_records(output_dir: Path) -> list[dict]:
    records = []
    for path in output_dir.glob("datasets/*/leiden/evaluation_seed_*.json"):
        records.append(json.loads(path.read_text()))
    for path in output_dir.glob("datasets/*/hdbscan/record.json"):
        records.append(json.loads(path.read_text()))
    return records


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    failures = []

    for dataset in args.datasets:
        dataset_dir = args.output_dir / "datasets" / dataset
        dataset_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n===== {dataset} =====", flush=True)
        try:
            scene = load_scene(dataset, data_dir=args.data_dir)
        except Exception as exc:
            failure = {
                "dataset": dataset,
                "method": "load_scene",
                "error": f"{type(exc).__name__}: {exc}",
            }
            failures.append(failure)
            print(json.dumps(failure), flush=True)
            continue

        cube = scene.cube.astype(np.float32, copy=False)
        # The loader returns GT, but neither method nor selector receives it.
        # It is passed only to add_evaluation after fitting/selection.
        gt = scene.gt.reshape(-1)
        for method in args.methods:
            try:
                if method == "leiden":
                    records = run_leiden(dataset, cube, gt, dataset_dir, args)
                    for record in records:
                        path = (
                            dataset_dir
                            / "leiden"
                            / f"evaluation_seed_{record['seed']}.json"
                        )
                        json_dump(path, record)
                        print(json.dumps(record), flush=True)
                else:
                    records = run_hdbscan(dataset, cube, gt, dataset_dir, args)
                    print(json.dumps(records[0]), flush=True)
            except Exception as exc:
                failure = {
                    "dataset": dataset,
                    "method": method,
                    "error": f"{type(exc).__name__}: {exc}",
                }
                failures.append(failure)
                json_dump(dataset_dir / method / "failure.json", failure)
                print(json.dumps(failure), flush=True)

        records = collect_existing_records(args.output_dir)
        write_csv(args.output_dir / "no_k_per_run.csv", records)
        write_csv(args.output_dir / "no_k_aggregate.csv", aggregate(records))

    records = collect_existing_records(args.output_dir)
    expected = len(args.datasets) * len(args.methods)
    completed_pairs = sorted({(row["dataset"], row["method"].lower()) for row in records})
    requested_pairs = {
        (dataset, method) for dataset in args.datasets for method in args.methods
    }
    completed_requested_pairs = sorted(set(completed_pairs) & requested_pairs)
    manifest = {
        "status": (
            "complete"
            if len(completed_requested_pairs) == expected and not failures
            else "partial"
        ),
        "datasets_requested": args.datasets,
        "methods_requested": args.methods,
        "completed_requested_dataset_method_pairs": len(completed_requested_pairs),
        "expected_dataset_method_pairs": expected,
        "all_completed_dataset_method_pairs_in_output": len(completed_pairs),
        "per_run_records": len(records),
        "failures": failures,
        "protocol": {
            "true_k_supplied": False,
            "labels_used_for_fitting_or_selection": False,
            "leiden_graph": (
                "exact frozen weighted mutual-kNN graph: z-score bands, "
                "PCA-20 neighbour search, k=20, radius=7, min_degree=4, "
                "Gaussian spectral-spatial edge affinity"
            ),
            "leiden_selector": (
                "fixed resolution grid; cross-seed and adjacent-resolution ARI; "
                "largest cluster <= 0.40; normalized entropy >= 0.35"
            ),
            "hdbscan": (
                "sklearn.cluster.HDBSCAN defaults on StandardScaler + PCA-20; "
                "noise retained as label -1 and reported"
            ),
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "sklearn": __import__("sklearn").__version__,
            "numpy": np.__version__,
            "igraph": getattr(__import__("igraph"), "__version__", "unknown"),
            "leidenalg": getattr(__import__("leidenalg"), "__version__", "unknown"),
        },
    }
    json_dump(args.output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
