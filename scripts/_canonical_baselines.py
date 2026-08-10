"""Canonical, auditable baseline benchmark for the journal comparison.

All methods cluster every scene pixel and are evaluated only on nonzero ground
truth pixels. The shared preprocessing is band standardization followed by
PCA-20. Baselines marked ``uses_true_k`` receive the reference class count;
Graph-RMS does not. Per-seed rows, aggregate statistics, wall time, and sampled
peak resident memory are written to disk.
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
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from graphrms.data import (DIRECT_MAT_SCENES, MAT_SCENES, SCENE_FILES,
                           URL_MAT_SCENES, load_scene)
from graphrms.metrics import evaluate


SCALABLE_METHODS = ("pca_kmeans", "minibatch_kmeans", "slic_kmeans")
OPTIONAL_METHODS = ("fcm",)
METRIC_KEYS = ("overall_accuracy", "balanced_accuracy", "kappa", "nmi", "ari")


class PeakRSS:
    def __init__(self, interval: float = 0.05):
        self.interval = interval
        self.peak = 0
        self._stop = threading.Event()
        self._thread = None

    def __enter__(self):
        process = psutil.Process(os.getpid())

        def sample():
            while not self._stop.is_set():
                self.peak = max(self.peak, process.memory_info().rss)
                self._stop.wait(self.interval)

        self._thread = threading.Thread(target=sample, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._stop.set()
        self._thread.join()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=[*MAT_SCENES, *SCENE_FILES, *DIRECT_MAT_SCENES,
                                          *URL_MAT_SCENES], default="indian_pines")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--output-dir", type=Path, default=Path("experiments/canonical_baselines"))
    p.add_argument("--methods", nargs="+", choices=[*SCALABLE_METHODS, *OPTIONAL_METHODS],
                   default=list(SCALABLE_METHODS))
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--pca-components", type=int, default=20)
    p.add_argument("--slic-region-size", type=int, default=100,
                   help="target pixels per SLIC region; fixed across datasets")
    p.add_argument("--fcm-max-iter", type=int, default=150)
    return p.parse_args()


def preprocess(spectra: np.ndarray, components: int) -> np.ndarray:
    x = StandardScaler(copy=True).fit_transform(spectra).astype(np.float32, copy=False)
    components = min(components, x.shape[1], x.shape[0] - 1)
    return PCA(n_components=components, svd_solver="randomized", random_state=0).fit_transform(x).astype(np.float32)


def slic_region_features(features: np.ndarray, h: int, w: int, target_size: int):
    from skimage.segmentation import slic

    image = features[:, :min(5, features.shape[1])].reshape(h, w, -1)
    lo = image.min(axis=(0, 1), keepdims=True)
    span = np.ptp(image, axis=(0, 1), keepdims=True)
    image = (image - lo) / np.maximum(span, 1e-8)
    n_segments = max(2, int(np.ceil(h * w / target_size)))
    regions = slic(
        image, n_segments=n_segments, compactness=0.1,
        channel_axis=-1, start_label=0, enforce_connectivity=True,
    ).reshape(-1)
    ids, inverse, counts = np.unique(regions, return_inverse=True, return_counts=True)
    sums = np.zeros((ids.size, features.shape[1]), dtype=np.float64)
    np.add.at(sums, inverse, features)
    return (sums / counts[:, None]).astype(np.float32), inverse, int(ids.size)


def fuzzy_cmeans(features: np.ndarray, n_clusters: int, seed: int, max_iter: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    u = rng.random((features.shape[0], n_clusters), dtype=np.float32)
    u /= u.sum(axis=1, keepdims=True)
    x2 = np.sum(features * features, axis=1, keepdims=True)
    for _ in range(max_iter):
        um = u * u
        centers = (um.T @ features) / np.maximum(um.sum(axis=0)[:, None], 1e-12)
        dist2 = np.maximum(x2 + np.sum(centers * centers, axis=1)[None, :] - 2 * features @ centers.T, 1e-12)
        new_u = 1.0 / dist2
        new_u /= new_u.sum(axis=1, keepdims=True)
        delta = float(np.max(np.abs(new_u - u)))
        u = new_u.astype(np.float32, copy=False)
        if delta < 1e-4:
            break
    return u.argmax(axis=1).astype(np.int32)


def run_method(method: str, features: np.ndarray, n_classes: int, seed: int,
               h: int, w: int, slic_cache, args: argparse.Namespace) -> np.ndarray:
    if method == "pca_kmeans":
        return KMeans(n_clusters=n_classes, n_init=20, random_state=seed).fit_predict(features)
    if method == "minibatch_kmeans":
        return MiniBatchKMeans(
            n_clusters=n_classes, n_init=20, batch_size=4096,
            max_iter=300, random_state=seed,
        ).fit_predict(features)
    if method == "slic_kmeans":
        region_features, pixel_to_region, _ = slic_cache
        region_labels = KMeans(
            n_clusters=n_classes, n_init=20, random_state=seed
        ).fit_predict(region_features)
        return region_labels[pixel_to_region]
    if method == "fcm":
        return fuzzy_cmeans(features, n_classes, seed, args.fcm_max_iter)
    raise ValueError(method)


def aggregate(rows: list[dict]) -> list[dict]:
    output = []
    for method in sorted({row["method"] for row in rows}):
        group = [row for row in rows if row["method"] == method]
        item = {"method": method, "n_repeats": len(group), "uses_true_k": True}
        for key in (*METRIC_KEYS, "runtime_seconds", "peak_rss_mb", "n_pred_clusters_labeled"):
            values = np.asarray([row[key] for row in group], dtype=float)
            item[f"{key}_mean"] = float(values.mean())
            item[f"{key}_std"] = float(values.std(ddof=1)) if values.size > 1 else 0.0
        output.append(item)
    return output


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scene = load_scene(args.dataset, data_dir=args.data_dir)
    h, w, bands = scene.cube.shape
    spectra = scene.cube.reshape(-1, bands).astype(np.float32, copy=False)
    gt = scene.gt.reshape(-1)
    n_classes = int(np.unique(gt[gt > 0]).size)

    t0 = time.perf_counter()
    with PeakRSS() as mem:
        features = preprocess(spectra, args.pca_components)
    preprocess_seconds = time.perf_counter() - t0
    preprocess_peak_rss_mb = mem.peak / 1e6
    slic_cache = slic_region_features(features, h, w, args.slic_region_size) if "slic_kmeans" in args.methods else None
    print(f"[baseline] {args.dataset}: N={gt.size}, bands={bands}, classes={n_classes}, "
          f"PCA={features.shape[1]} ({preprocess_seconds:.2f}s)", flush=True)

    rows = []
    for method in args.methods:
        for seed in args.seeds:
            start = time.perf_counter()
            with PeakRSS() as method_mem:
                labels = run_method(method, features, n_classes, seed, h, w, slic_cache, args)
                metrics = evaluate(gt, labels)
            row = {
                "dataset": args.dataset,
                "method": method,
                "seed": seed,
                "uses_true_k": True,
                "runtime_seconds": time.perf_counter() - start,
                "peak_rss_mb": method_mem.peak / 1e6,
                **{key: metrics[key] for key in METRIC_KEYS},
                "n_pred_clusters_labeled": metrics["n_pred_clusters"],
            }
            rows.append(row)
            print(json.dumps(row), flush=True)

    aggregate_rows = aggregate(rows)
    with (args.output_dir / f"{args.dataset}_per_seed.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)
    with (args.output_dir / f"{args.dataset}_summary.json").open("w") as f:
        json.dump({
            "dataset": args.dataset,
            "protocol": {
                "cluster_scope": "all scene pixels",
                "evaluation_scope": "ground-truth labels > 0",
                "shared_preprocessing": "per-band StandardScaler then randomized PCA",
                "pca_components": features.shape[1],
                "preprocessing_random_seed": 0,
                "baselines_receive_true_class_count": True,
                "seeds": args.seeds,
                "slic_target_pixels_per_region": args.slic_region_size,
            },
            "scene": {"height": h, "width": w, "bands": bands, "n_classes": n_classes},
            "preprocess_seconds": preprocess_seconds,
            "preprocess_peak_rss_mb": preprocess_peak_rss_mb,
            "slic_regions": slic_cache[2] if slic_cache is not None else None,
            "aggregate": aggregate_rows,
            "per_seed": rows,
        }, f, indent=2)


if __name__ == "__main__":
    main()
