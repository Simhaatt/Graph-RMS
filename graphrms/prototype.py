"""Experimental second-stage consolidation of a pure fine Graph-RMS partition.

Each fine cluster becomes a mode prototype in standardized diffusion-PCA space.
Prototype distances are divided by their pooled within-cluster dispersion,
making the merge scale dimensionless and potentially more transferable than a
single absolute pixel-level radius. This is a development experiment; reference
labels are used only to audit the resulting curve.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from graphrms.affinity import compute_gaussian_affinities
from graphrms.data import load_scene
from graphrms.graph import build_graph
from graphrms.metrics import evaluate
from graphrms.postprocess import majority_filter
from graphrms.rms import compute_merge_neighbors, merge_from_neighbors, run_graph_rms


FINE_SETTINGS = {
    "salinas_a": (50, 0.40),
    "indian_pines": (250, 0.25),
    "pavia_university": (1000, 0.10),
    "whu_hi_honghu": (250, 0.10),
    "whu_hi_longkou": (750, 0.25),
    "whu_hi_hanchuan": (100, 0.10),
}

# The validation notebook always overrides these placeholders with its
# label-free surface selections. They are intentionally outside frozen v1.
VALIDATION_PLACEHOLDER_SETTINGS = {
    "ksc": (250, 0.10), "botswana": (250, 0.10), "houston13": (100, 0.10),
    "trento": (100, 0.10)
}

PROTOTYPE_THRESHOLDS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=[*FINE_SETTINGS, *VALIDATION_PLACEHOLDER_SETTINGS],
                   default="indian_pines")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--output-dir", type=Path, default=Path("experiments/prototype_consolidation"))
    p.add_argument("--checkpoint", type=int, default=None)
    p.add_argument("--fine-radius", type=float, default=None)
    p.add_argument("--prototype-neighbors", type=int, default=30)
    p.add_argument("--linkage", choices=["single", "average", "complete", "reciprocal"],
                   default="complete")
    p.add_argument("--size-exponent", type=float, default=0.0,
                   help="exploratory confidence weighting: dispersion is divided by "
                        "(mode_size / median_size)^exponent; v1 uses 0")
    p.add_argument("--selector-version", type=int, choices=[1, 2, 3], default=1,
                   help="v1 is frozen; v2 is persistent-plateau; v3 is compression-aware")
    p.add_argument("--thresholds", type=float, nargs="+",
                   default=PROTOTYPE_THRESHOLDS)
    p.add_argument("--spectral-preprocess", choices=["raw", "band_zscore"], default="raw",
                   help="label-free spectral preprocessing before graph construction and RMS")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def standardized_diffusion_space(y_final: np.ndarray, components: int = 10, seed: int = 0) -> np.ndarray:
    n_components = min(components, y_final.shape[1], y_final.shape[0] - 1)
    reduced = PCA(n_components=n_components, random_state=seed).fit_transform(y_final)
    std = reduced.std(axis=0)
    std[std < 1e-8] = 1.0
    return ((reduced - reduced.mean(axis=0)) / std).astype(np.float32)


def mode_prototypes(features: np.ndarray, labels: np.ndarray):
    ids, inverse, counts = np.unique(labels, return_inverse=True, return_counts=True)
    membership = sparse.csr_matrix(
        (np.ones(labels.size, dtype=np.float32), (inverse, np.arange(labels.size))),
        shape=(ids.size, labels.size),
    )
    means = np.asarray(membership @ features, dtype=np.float64) / counts[:, None]
    second = np.asarray(membership @ (features * features), dtype=np.float64) / counts[:, None]
    variance = np.maximum(second - means * means, 0).sum(axis=1)
    dispersion = np.sqrt(variance)
    positive = dispersion[(dispersion > 1e-6) & (counts >= 5)]
    floor = float(np.percentile(positive, 25)) if positive.size else 1e-3
    dispersion = np.maximum(dispersion, floor)
    return means.astype(np.float32), dispersion.astype(np.float32), inverse, counts, floor


def adjusted_dispersions(dispersions: np.ndarray, counts: np.ndarray | None,
                         size_exponent: float) -> np.ndarray:
    if size_exponent == 0 or counts is None:
        return dispersions
    relative_size = counts / max(float(np.median(counts)), 1.0)
    return dispersions / np.maximum(relative_size, 1e-8) ** size_exponent


def prototype_edges(means: np.ndarray, dispersions: np.ndarray, k: int,
                    counts: np.ndarray | None = None, size_exponent: float = 0.0):
    n = means.shape[0]
    k = min(k + 1, n)
    nn = NearestNeighbors(n_neighbors=k).fit(means)
    distances, indices = nn.kneighbors(means)
    rows = np.repeat(np.arange(n, dtype=np.int64), k - 1)
    cols = indices[:, 1:].reshape(-1)
    raw = distances[:, 1:].reshape(-1)
    effective = adjusted_dispersions(dispersions, counts, size_exponent)
    pooled = np.sqrt(effective[rows] ** 2 + effective[cols] ** 2)
    normalized = raw / np.maximum(pooled, 1e-8)
    return rows, cols, normalized


def prototype_distance_matrix(means: np.ndarray, dispersions: np.ndarray,
                              counts: np.ndarray | None = None,
                              size_exponent: float = 0.0) -> np.ndarray:
    raw = pairwise_distances(means, metric="euclidean")
    effective = adjusted_dispersions(dispersions, counts, size_exponent)
    pooled = np.sqrt(effective[:, None] ** 2 + effective[None, :] ** 2)
    normalized = raw / np.maximum(pooled, 1e-8)
    np.fill_diagonal(normalized, 0.0)
    return normalized


def consolidate(n_modes: int, rows: np.ndarray, cols: np.ndarray,
                normalized_distance: np.ndarray, threshold: float,
                linkage: str, distance_matrix: np.ndarray | None = None) -> np.ndarray:
    if threshold <= 0:
        return np.arange(n_modes, dtype=np.int32)
    if linkage != "single":
        if distance_matrix is None:
            raise ValueError("distance_matrix is required for average/complete linkage")
        model = AgglomerativeClustering(
            n_clusters=None,
            metric="precomputed",
            linkage=linkage,
            distance_threshold=threshold,
            compute_full_tree=True,
        )
        return model.fit_predict(distance_matrix).astype(np.int32)
    keep = normalized_distance <= threshold
    graph = sparse.coo_matrix(
        (np.ones(int(keep.sum()), dtype=np.uint8), (rows[keep], cols[keep])),
        shape=(n_modes, n_modes),
    )
    _, labels = connected_components(graph, directed=False)
    return labels.astype(np.int32)


def reciprocal_consolidate(means: np.ndarray, dispersions: np.ndarray,
                           counts: np.ndarray, threshold: float, k: int = 30,
                           size_exponent: float = 0.0) -> np.ndarray:
    """Agglomerate reciprocal nearest prototypes without single-link chaining.

    Only mutually preferred pairs merge in a round. Centroids and pooled
    within-mode dispersion are then recomputed before another round, so a
    bridge of individually short edges cannot collapse an entire component in
    one operation. The returned labels index the original fine modes.
    """
    n_original = means.shape[0]
    if threshold <= 0 or n_original < 2:
        return np.arange(n_original, dtype=np.int32)
    current_means = means.astype(np.float64, copy=True)
    current_disp = dispersions.astype(np.float64, copy=True)
    current_counts = counts.astype(np.float64, copy=True)
    original_to_current = np.arange(n_original, dtype=np.int32)

    while current_means.shape[0] > 1:
        n_current = current_means.shape[0]
        n_neighbors = min(max(2, k + 1), n_current)
        distances, indices = NearestNeighbors(n_neighbors=n_neighbors).fit(
            current_means
        ).kneighbors(current_means)
        candidate_idx = indices[:, 1:]
        candidate_raw = distances[:, 1:]
        effective = adjusted_dispersions(current_disp, current_counts, size_exponent)
        pooled = np.sqrt(
            effective[:, None] ** 2 + effective[candidate_idx] ** 2
        )
        normalized = candidate_raw / np.maximum(pooled, 1e-8)
        best_position = np.argmin(normalized, axis=1)
        best = candidate_idx[np.arange(n_current), best_position]
        best_distance = normalized[np.arange(n_current), best_position]
        pairs = [
            (i, int(j)) for i, j in enumerate(best)
            if i < j and best[int(j)] == i
            and best_distance[i] <= threshold and best_distance[int(j)] <= threshold
        ]
        if not pairs:
            break

        partner = np.full(n_current, -1, dtype=np.int32)
        for left, right in pairs:
            partner[left] = right
            partner[right] = left
        new_index = np.full(n_current, -1, dtype=np.int32)
        new_means, new_disp, new_counts = [], [], []
        for i in range(n_current):
            if new_index[i] >= 0:
                continue
            j = int(partner[i])
            members = [i] if j < 0 else [i, j]
            total = float(current_counts[members].sum())
            centroid = np.sum(
                current_means[members] * current_counts[members, None], axis=0
            ) / total
            pooled_variance = sum(
                current_counts[m] * (
                    current_disp[m] ** 2
                    + float(np.sum((current_means[m] - centroid) ** 2))
                )
                for m in members
            ) / total
            idx = len(new_means)
            for m in members:
                new_index[m] = idx
            new_means.append(centroid)
            new_disp.append(np.sqrt(max(float(pooled_variance), 0.0)))
            new_counts.append(total)
        original_to_current = new_index[original_to_current]
        current_means = np.asarray(new_means, dtype=np.float64)
        current_disp = np.asarray(new_disp, dtype=np.float64)
        current_counts = np.asarray(new_counts, dtype=np.float64)

    return original_to_current.astype(np.int32)


def partition_stats(labels: np.ndarray) -> dict:
    _, counts = np.unique(labels, return_counts=True)
    probs = counts / counts.sum()
    entropy = -float(np.sum(probs * np.log(probs + 1e-12)))
    return {
        "n_clusters_full": int(counts.size),
        "largest_cluster_fraction_full": float(counts.max() / counts.sum()),
        "normalized_entropy_full": entropy / max(float(np.log(counts.size)), 1e-12) if counts.size > 1 else 0.0,
        "microcluster_mass_full": float(counts[counts < 20].sum() / counts.sum()),
    }


def select_consolidation(rows: list[dict], n_pixels: int) -> tuple[dict, dict]:
    """Choose a non-degenerate stable endpoint without reference labels."""
    baseline = rows[0]
    cluster_gate = max(20, int(np.ceil(0.5 * np.sqrt(n_pixels))))
    gate_passed = baseline["n_clusters_full"] > cluster_gate
    audit = {
        "fragmentation_cluster_gate": cluster_gate,
        "fragmentation_gate_passed": gate_passed,
        "rules": {
            "adjacent_partition_ari_min": 0.95,
            "largest_cluster_fraction_max": 0.40,
            "normalized_entropy_min": 0.35,
        },
    }
    if not gate_passed:
        return baseline, {**audit, "reason": "fine partition is not fragmented"}

    eligible = []
    plateau_started = False
    for row in rows[1:]:
        passes = (
            row["adjacent_partition_ari_label_free"] >= 0.95
            and row["largest_cluster_fraction_full"] <= 0.40
            and row["normalized_entropy_full"] >= 0.35
        )
        if passes:
            eligible.append(row)
            plateau_started = True
        elif plateau_started:
            break
    if not eligible:
        return baseline, {**audit, "reason": "no stable non-degenerate consolidation plateau"}
    return eligible[-1], {**audit, "reason": "largest threshold in first stable non-degenerate plateau"}


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    default_t, default_radius = (FINE_SETTINGS | VALIDATION_PLACEHOLDER_SETTINGS)[args.dataset]
    checkpoint = args.checkpoint if args.checkpoint is not None else default_t
    fine_radius = args.fine_radius if args.fine_radius is not None else default_radius
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu") if args.cpu else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    total_start = time.perf_counter()
    stage_seconds = {}

    stage_start = time.perf_counter()
    scene = load_scene(args.dataset, data_dir=args.data_dir)
    cube = scene.cube.astype(np.float32, copy=False)
    h, w, bands = cube.shape
    if args.spectral_preprocess == "band_zscore":
        flat = cube.reshape(-1, bands)
        mean = flat.mean(axis=0, dtype=np.float64).astype(np.float32)
        std = flat.std(axis=0, dtype=np.float64).astype(np.float32)
        std[std < 1e-8] = 1.0
        cube = ((cube - mean) / std).astype(np.float32)
    spectra = cube.reshape(-1, bands)
    gt = scene.gt.reshape(-1)
    graph = build_graph(cube, pca_components=20, k=20, window_radius=7,
                        mutual=True, min_degree=4)
    stage_seconds["load_and_graph"] = time.perf_counter() - stage_start
    stage_start = time.perf_counter()
    affinity, _ = compute_gaussian_affinities(
        spectra, graph.edge_i, graph.edge_j, graph.spatial_dist, seed=args.seed
    )
    stage_seconds["affinity"] = time.perf_counter() - stage_start
    stage_start = time.perf_counter()
    y_final, _, _ = run_graph_rms(
        spectra, graph.edge_i, graph.edge_j, affinity, graph.n_nodes,
        max_iter=checkpoint, tol=0.0, alpha=0.5, device=device,
    )
    if device.type == "cuda":
        torch.cuda.synchronize()
    stage_seconds["diffusion"] = time.perf_counter() - stage_start
    stage_start = time.perf_counter()
    fine_neighbors = compute_merge_neighbors(y_final, pca_components=10, device=device)
    fine_raw, _ = merge_from_neighbors(fine_neighbors, tol=fine_radius)
    fine_labels = majority_filter(fine_raw, h, w, radius=3)

    features = standardized_diffusion_space(y_final, seed=args.seed)
    means, dispersions, pixel_to_mode, mode_sizes, dispersion_floor = mode_prototypes(features, fine_labels)
    edge_i, edge_j, normalized_distance = prototype_edges(
        means, dispersions, args.prototype_neighbors, mode_sizes, args.size_exponent
    )
    if args.linkage in {"single", "reciprocal"}:
        distance_matrix = None
        distance_sample = normalized_distance
    else:
        distance_matrix = prototype_distance_matrix(means, dispersions, mode_sizes, args.size_exponent)
        distance_sample = distance_matrix[np.triu_indices(means.shape[0], k=1)]
    print(f"[prototype] fine modes={means.shape[0]}, dispersion_floor={dispersion_floor:.5f}")
    print("[prototype] normalized distance percentiles:",
          np.percentile(distance_sample, [1, 10, 25, 50, 75, 90, 99]).round(4).tolist())

    rows_out = []
    previous = None
    from sklearn.metrics import adjusted_rand_score
    for threshold in sorted(set(args.thresholds)):
        if args.linkage == "reciprocal":
            mode_groups = reciprocal_consolidate(
                means, dispersions, mode_sizes, threshold,
                args.prototype_neighbors, args.size_exponent,
            )
        else:
            mode_groups = consolidate(
                means.shape[0], edge_i, edge_j, normalized_distance, threshold,
                args.linkage, distance_matrix,
            )
        labels = mode_groups[pixel_to_mode]
        metrics = evaluate(gt, labels)
        row = {
            "prototype_threshold": threshold,
            **partition_stats(labels),
            "adjacent_partition_ari_label_free": (
                float(adjusted_rand_score(previous, labels)) if previous is not None else None
            ),
            "overall_accuracy": metrics["overall_accuracy"],
            "balanced_accuracy": metrics["balanced_accuracy"],
            "kappa": metrics["kappa"],
            "nmi": metrics["nmi"],
            "ari": metrics["ari"],
            "n_pred_clusters_labeled": metrics["n_pred_clusters"],
        }
        rows_out.append(row)
        previous = labels
        print(json.dumps(row), flush=True)

    if args.selector_version == 1:
        selected, selection_audit = select_consolidation(rows_out, gt.size)
    elif args.selector_version == 2:
        from experiments.select_prototype_scale_v2 import select_v2
        selected, selection_audit = select_v2(rows_out, gt.size)
    else:
        from experiments.select_prototype_scale_v3 import select_v3
        selected, selection_audit = select_v3(rows_out, gt.size)
    if args.linkage == "reciprocal":
        selected_groups = reciprocal_consolidate(
            means, dispersions, mode_sizes, selected["prototype_threshold"],
            args.prototype_neighbors, args.size_exponent,
        )
    else:
        selected_groups = consolidate(
            means.shape[0], edge_i, edge_j, normalized_distance,
            selected["prototype_threshold"], args.linkage, distance_matrix,
        )
    selected_labels = selected_groups[pixel_to_mode]
    np.save(args.output_dir / f"{args.dataset}_selected_labels.npy", selected_labels.reshape(h, w))
    stage_seconds["fine_merge_and_prototype_consolidation"] = time.perf_counter() - stage_start
    print("[prototype] label-free selection:", json.dumps(selected), flush=True)

    with (args.output_dir / f"{args.dataset}_curve.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows_out[0].keys())
        writer.writeheader()
        writer.writerows(rows_out)
    with (args.output_dir / f"{args.dataset}_summary.json").open("w") as f:
        json.dump({
            "dataset": args.dataset,
            "seed": args.seed,
            "fine_checkpoint": checkpoint,
            "fine_radius": fine_radius,
            "spectral_preprocess": args.spectral_preprocess,
            "fine_modes": int(means.shape[0]),
            "dispersion_floor": dispersion_floor,
            "prototype_neighbors": args.prototype_neighbors,
            "size_exponent": args.size_exponent,
            "selector_version": args.selector_version,
            "linkage": args.linkage,
            "label_free_selected": selected,
            "selection_audit": selection_audit,
            "stage_runtime_seconds": stage_seconds,
            "total_runtime_seconds": time.perf_counter() - total_start,
            "peak_gpu_memory_mb": (
                float(torch.cuda.max_memory_allocated() / 1e6) if device.type == "cuda" else None
            ),
            "curve": rows_out,
        }, f, indent=2)


if __name__ == "__main__":
    main()
