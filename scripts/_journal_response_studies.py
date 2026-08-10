"""Journal-response ablation, sensitivity, and resource studies for adaptive-v5.

The frozen final configuration is read from
``results/provenance/frozen_results_registry.json``. Ground-truth labels
are used only after a partition has been produced. Each one-factor study keeps
the frozen prototype threshold and all non-varied settings unchanged.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import threading
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from graphrms.prototype import (
    mode_prototypes,
    partition_stats,
    reciprocal_consolidate,
    standardized_diffusion_space,
)
from graphrms.affinity import compute_gaussian_affinities
from graphrms.data import load_scene
from graphrms.graph import build_graph
from graphrms.metrics import evaluate
from graphrms.postprocess import majority_filter
from graphrms.rms import compute_merge_neighbors, merge_from_neighbors, run_graph_rms


REGISTRY = ROOT / "results" / "provenance" / "frozen_results_registry.json"


@dataclass(frozen=True)
class Configuration:
    pca_components: int = 20
    k: int = 20
    window_radius: int = 7
    mutual: bool = True
    min_degree: int = 4
    use_spatial_affinity: bool = True
    alpha: float = 0.5
    checkpoint: int = 100
    fine_radius: float = 0.15
    filter_radius: int = 3
    prototype_threshold: float = 0.75
    prototype_neighbors: int = 30
    size_exponent: float = 0.5
    disable_fine_grouping: bool = False
    disable_consolidation: bool = False


class PeakRSSMonitor:
    """Sample process RSS while a configuration executes."""

    def __init__(self, interval: float = 0.05):
        self.interval = interval
        self.peak = 0
        self._done = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self):
        try:
            import psutil
        except ImportError:
            return self
        process = psutil.Process(os.getpid())

        def sample():
            while not self._done.is_set():
                try:
                    self.peak = max(self.peak, int(process.memory_info().rss))
                except Exception:
                    break
                self._done.wait(self.interval)

        self._thread = threading.Thread(target=sample, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._done.set()
        if self._thread is not None:
            self._thread.join(timeout=1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", choices=["ablation", "sensitivity", "profile", "all"],
                        default="all")
    parser.add_argument("--datasets", nargs="+", default=["salinas_a", "indian_pines"])
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("journal_validation/final_v5_studies"))
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--save-labels", action="store_true")
    parser.add_argument("--profile-repeats", type=int, default=3)
    parser.add_argument("--fast", action="store_true",
                        help="Use three-point sensitivity grids for a quick local check.")
    return parser.parse_args()


def load_registry() -> dict:
    return json.loads(REGISTRY.read_text())


def frozen_configuration(dataset: str, registry: dict) -> Configuration:
    row = registry["datasets"][dataset]
    return Configuration(
        checkpoint=int(row["fine_checkpoint"]),
        fine_radius=float(row["fine_radius"]),
        prototype_threshold=float(row["prototype_threshold"]),
        size_exponent=float(row["size_exponent"]),
    )


def band_zscore(cube: np.ndarray) -> np.ndarray:
    flat = cube.reshape(-1, cube.shape[-1])
    mean = flat.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = flat.std(axis=0, dtype=np.float64).astype(np.float32)
    std[std < 1e-8] = 1.0
    return ((cube - mean) / std).astype(np.float32)


def run_configuration(
    dataset: str,
    config: Configuration,
    variant: str,
    study: str,
    data_dir: str,
    output_dir: Path,
    device: torch.device,
    save_labels: bool,
) -> dict:
    np.random.seed(0)
    torch.manual_seed(0)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    stages: dict[str, float] = {}
    total_start = time.perf_counter()

    with PeakRSSMonitor() as rss:
        start = time.perf_counter()
        scene = load_scene(dataset, data_dir=data_dir)
        cube = band_zscore(scene.cube.astype(np.float32, copy=False))
        h, w, bands = cube.shape
        spectra = cube.reshape(-1, bands)
        gt = scene.gt.reshape(-1)
        graph = build_graph(
            cube,
            pca_components=config.pca_components,
            k=config.k,
            window_radius=config.window_radius,
            mutual=config.mutual,
            min_degree=config.min_degree,
        )
        stages["load_preprocess_graph_seconds"] = time.perf_counter() - start

        start = time.perf_counter()
        affinities, bandwidths = compute_gaussian_affinities(
            spectra,
            graph.edge_i,
            graph.edge_j,
            graph.spatial_dist,
            seed=0,
            use_spatial=config.use_spatial_affinity,
        )
        stages["affinity_seconds"] = time.perf_counter() - start

        start = time.perf_counter()
        y_final, _, movement = run_graph_rms(
            spectra,
            graph.edge_i,
            graph.edge_j,
            affinities,
            graph.n_nodes,
            max_iter=config.checkpoint,
            tol=0.0,
            alpha=config.alpha,
            device=device,
        )
        if device.type == "cuda":
            torch.cuda.synchronize()
        stages["diffusion_seconds"] = time.perf_counter() - start

        start = time.perf_counter()
        if config.disable_fine_grouping:
            fine_raw = np.arange(graph.n_nodes, dtype=np.int32)
        else:
            fine_neighbors = compute_merge_neighbors(
                y_final, pca_components=10, device=device
            )
            fine_raw, _ = merge_from_neighbors(
                fine_neighbors, tol=config.fine_radius
            )
        fine_labels = (
            fine_raw
            if config.filter_radius == 0
            else majority_filter(fine_raw, h, w, radius=config.filter_radius)
        )
        fine_metrics = evaluate(gt, fine_labels)

        features = standardized_diffusion_space(y_final, seed=0)
        means, dispersions, pixel_to_mode, mode_sizes, dispersion_floor = (
            mode_prototypes(features, fine_labels)
        )
        threshold = 0.0 if config.disable_consolidation else config.prototype_threshold
        mode_groups = reciprocal_consolidate(
            means,
            dispersions,
            mode_sizes,
            threshold,
            k=config.prototype_neighbors,
            size_exponent=config.size_exponent,
        )
        labels = mode_groups[pixel_to_mode]
        stages["mode_extraction_consolidation_seconds"] = time.perf_counter() - start

        metrics = evaluate(gt, labels)
        full_stats = partition_stats(labels)
        fine_stats = partition_stats(fine_labels)
        total_seconds = time.perf_counter() - total_start

    degrees = np.diff(graph.indptr)
    record = {
        "study": study,
        "dataset": dataset,
        "variant": variant,
        **asdict(config),
        "device": str(device),
        "height": h,
        "width": w,
        "bands": bands,
        "n_nodes": int(graph.n_nodes),
        "directed_edge_entries": int(graph.edge_i.size),
        "degree_min": int(degrees.min()),
        "degree_median": float(np.median(degrees)),
        "degree_p90": float(np.percentile(degrees, 90)),
        "degree_max": int(degrees.max()),
        "isolated_nodes": int(np.sum(degrees == 0)),
        "sigma_spectral": float(bandwidths.sigma_spectral),
        "sigma_spatial": float(bandwidths.sigma_spatial),
        "last_movement": (
            float(movement[-1]) if isinstance(movement, (list, tuple, np.ndarray))
            and len(movement) else float(movement or 0.0)
        ),
        "fine_modes_full": int(fine_stats["n_clusters_full"]),
        "fine_oa": float(fine_metrics["overall_accuracy"]),
        "fine_balanced_accuracy": float(fine_metrics["balanced_accuracy"]),
        "fine_nmi": float(fine_metrics["nmi"]),
        "fine_ari": float(fine_metrics["ari"]),
        "dispersion_floor": float(dispersion_floor),
        **full_stats,
        "overall_accuracy": float(metrics["overall_accuracy"]),
        "balanced_accuracy": float(metrics["balanced_accuracy"]),
        "nmi": float(metrics["nmi"]),
        "ari": float(metrics["ari"]),
        "n_pred_clusters_labeled": int(metrics["n_pred_clusters"]),
        **stages,
        "total_runtime_seconds": float(total_seconds),
        "peak_process_rss_mb": float(rss.peak / 1e6) if rss.peak else None,
        "peak_gpu_allocated_mb": (
            float(torch.cuda.max_memory_allocated() / 1e6)
            if device.type == "cuda" else None
        ),
        "peak_gpu_reserved_mb": (
            float(torch.cuda.max_memory_reserved() / 1e6)
            if device.type == "cuda" else None
        ),
    }
    if save_labels:
        label_dir = output_dir / "labels"
        label_dir.mkdir(parents=True, exist_ok=True)
        np.save(label_dir / f"{dataset}__{study}__{variant}.npy",
                labels.reshape(h, w).astype(np.int32))
    return record


def ablation_configurations(full: Configuration) -> list[tuple[str, Configuration]]:
    return [
        ("full", full),
        ("graph_no_mutual_gate", replace(full, mutual=False, min_degree=0)),
        ("graph_no_spatial_affinity", replace(full, use_spatial_affinity=False)),
        ("diffusion_T0", replace(full, checkpoint=0)),
        ("mode_no_spatial_cleanup", replace(full, filter_radius=0)),
        ("mode_no_fine_grouping", replace(full, disable_fine_grouping=True,
                                          filter_radius=0)),
        ("consolidation_disabled", replace(full, disable_consolidation=True)),
        ("consolidation_no_size_weighting", replace(full, size_exponent=0.0)),
    ]


def sensitivity_configurations(
    full: Configuration, fast: bool
) -> list[tuple[str, Configuration]]:
    if fast:
        grids = {
            "window_radius": [3, 7, 9],
            "pca_components": [10, 20, 40],
            "checkpoint": sorted({max(1, full.checkpoint // 2),
                                  full.checkpoint, full.checkpoint * 2}),
            "fine_radius": [full.fine_radius * 0.75, full.fine_radius,
                            full.fine_radius * 1.25],
        }
    else:
        grids = {
            "window_radius": [3, 5, 7, 9],
            "pca_components": [10, 20, 30, 40],
            "checkpoint": sorted({max(1, full.checkpoint // 4),
                                  max(1, full.checkpoint // 2),
                                  full.checkpoint, full.checkpoint * 2}),
            "fine_radius": [full.fine_radius * factor
                            for factor in (0.5, 0.75, 1.0, 1.25, 1.5)],
        }
    result: list[tuple[str, Configuration]] = []
    for parameter, values in grids.items():
        for value in values:
            value = round(value, 8) if isinstance(value, float) else value
            result.append((f"{parameter}={value:g}",
                           replace(full, **{parameter: value})))
    return result


def write_records(records: list[dict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if not records:
        return
    json_path = output_dir / "journal_response_studies.json"
    json_path.write_text(json.dumps(records, indent=2))
    keys = list(records[0])
    for row in records[1:]:
        for key in row:
            if key not in keys:
                keys.append(key)
    with (output_dir / "journal_response_studies.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(records)


def main() -> None:
    args = parse_args()
    registry = load_registry()
    unknown = sorted(set(args.datasets) - set(registry["datasets"]))
    if unknown:
        raise ValueError(f"datasets absent from frozen registry: {unknown}")
    device = torch.device(
        "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    existing_json = args.output_dir / "journal_response_studies.json"
    records: list[dict] = (
        json.loads(existing_json.read_text()) if existing_json.exists() else []
    )
    completed = {
        (row["study"], row["dataset"], row["variant"]) for row in records
    }
    for dataset in args.datasets:
        full = frozen_configuration(dataset, registry)
        jobs: list[tuple[str, str, Configuration]] = []
        if args.study in {"ablation", "all"}:
            jobs.extend(("ablation", name, cfg)
                        for name, cfg in ablation_configurations(full))
        if args.study in {"sensitivity", "all"}:
            jobs.extend(("sensitivity", name, cfg)
                        for name, cfg in sensitivity_configurations(full, args.fast))
        if args.study in {"profile", "all"}:
            jobs.extend(("profile", f"repeat_{i + 1}", full)
                        for i in range(args.profile_repeats))
        for study, variant, config in jobs:
            if (study, dataset, variant) in completed:
                print(f"[resume-skip] {dataset} | {study} | {variant}", flush=True)
                continue
            print(f"[journal-response] {dataset} | {study} | {variant}", flush=True)
            record = run_configuration(
                dataset, config, variant, study, args.data_dir,
                args.output_dir, device, args.save_labels,
            )
            records.append(record)
            completed.add((study, dataset, variant))
            write_records(records, args.output_dir)
            print(json.dumps({
                "OA": record["overall_accuracy"],
                "BA": record["balanced_accuracy"],
                "NMI": record["nmi"],
                "ARI": record["ari"],
                "Khat": record["n_clusters_full"],
                "seconds": record["total_runtime_seconds"],
                "rss_mb": record["peak_process_rss_mb"],
                "gpu_mb": record["peak_gpu_allocated_mb"],
            }), flush=True)
    manifest = {
        "protocol": "frozen adaptive-v5 journal-response studies",
        "ground_truth_used_only_for_post-partition_evaluation": True,
        "datasets": args.datasets,
        "study": args.study,
        "device": str(device),
        "records": len(records),
        "output_csv": "journal_response_studies.csv",
        "output_json": "journal_response_studies.json",
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
