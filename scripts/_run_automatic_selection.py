"""Archived automatic-scale-v1 candidate generation for Graph-RMS.

This implementation generates the label-free endpoint surfaces consumed by
automatic-v2. Its historical v1 endpoint decision is retained for provenance
and backward compatibility but is not the reported automatic-v2 selector.
Reference labels are loaded only by the separate ``evaluate`` command.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.io import loadmat
from sklearn.metrics import adjusted_rand_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from graphrms.affinity import compute_gaussian_affinities
from graphrms.data import (DIRECT_MAT_SCENES, MAT_SCENES, REPO_ID, SCENE_FILES,
                           URL_MAT_SCENES, _read_envi_bsq, load_scene)
from graphrms.graph import build_graph
from graphrms.metrics import evaluate
from graphrms.postprocess import majority_filter
from graphrms.rms import (_row_normalized_weight_matrix, compute_merge_neighbors,
                          merge_from_neighbors)
from graphrms.prototype import (mode_prototypes, partition_stats,
                                reciprocal_consolidate,
                                standardized_diffusion_space)

DEVELOPMENT = ["salinas_a", "indian_pines", "ksc", "pavia_university",
               "whu_hi_longkou", "whu_hi_honghu", "whu_hi_hanchuan", "botswana"]
HOLDOUT = "trento"
ALL_DATASETS = DEVELOPMENT + [HOLDOUT]

# These values are protocol constants, not fitted to reference labels.
PROTOCOL = {
    "version": "automatic-scale-v1",
    "checkpoints": [25, 50, 100, 200],
    "scale_neighbor_q": 10,
    "gamma_grid": [0.75, 1.0, 1.25, 1.5],
    "prototype_thresholds": [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35,
                             0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75,
                             1.0, 1.5, 2.0, 3.0],
    "size_exponents": [0.0, 0.5],
    "prototype_neighbors": 30,
    "largest_cluster_fraction_max": 0.40,
    "normalized_entropy_min": 0.35,
    "compression_min": 0.20,
    "partition_ari_min": 0.95,
    "cross_scale_ari_min": 0.90,
    "minimum_tau_run": 1,
    "graph": {"pca_components": 20, "k": 20, "window_radius": 7,
              "mutual": True, "min_degree": 4},
    "diffusion": {"alpha": 0.5, "tol": 0.0},
    "fine_merge": {"pca_components": 10, "max_neighbors": 50,
                   "post_filter_radius": 3},
    "spectral_preprocess": "band_zscore",
    "selection_hierarchy": {
        "1_local_endpoint": "Within each (T,gamma,beta), retain candidates passing L/E/C and tau-adjacent ARI; choose minimum region count, then higher tau stability, then smaller tau.",
        "2_cross_scale": "Require endpoint ARI >= 0.90 with the same-beta endpoint at one adjacent T or gamma.",
        "3_checkpoint": "Retain the earliest T having at least one cross-scale-admitted endpoint.",
        "4_beta": "Use beta=0.5 when any size-aware endpoint is admitted at that T; otherwise fall back to beta=0.",
        "5_final": "Choose minimum region count, then higher cross-scale ARI, then smaller gamma, then smaller tau.",
        "6_abstain": "Return abstain if no endpoint passes every gate.",
    },
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _hf(repo: str, filename: str, data_dir: Path) -> Path:
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download(repo_id=repo, repo_type="dataset", filename=filename,
                                local_dir=str(data_dir)))


def _only_array(path: Path, ndim: int, key: str | None = None) -> np.ndarray:
    arrays = loadmat(path)
    if key is not None:
        return arrays[key]
    candidates = [v for k, v in arrays.items() if not k.startswith("__")
                  and isinstance(v, np.ndarray) and v.ndim == ndim]
    if len(candidates) != 1:
        raise ValueError(f"expected one {ndim}D array in {path}, found {len(candidates)}")
    return candidates[0]


def _align_known(cube: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    # Public MATLAB mirrors already use (H, W, bands).  Check this before
    # permutation discovery, which is intentionally ambiguous when H == W.
    if cube.shape[:2] == (h, w):
        return cube
    matches = []
    for ha in range(3):
        for wa in range(3):
            if ha != wa and cube.shape[ha] == h and cube.shape[wa] == w:
                ba = next(a for a in range(3) if a not in (ha, wa))
                matches.append((ha, wa, ba))
    if len(matches) != 1:
        raise ValueError(f"cannot uniquely align cube {cube.shape} to declared shape {shape}")
    return np.transpose(cube, matches[0])


def load_cube_only(dataset: str, data_dir: Path, shape_registry: dict) -> np.ndarray:
    """Load no reference-label file. Spatial dimensions come from public metadata."""
    data_dir.mkdir(parents=True, exist_ok=True)
    if dataset in MAT_SCENES:
        repo, cube_file, cube_key, _, _ = MAT_SCENES[dataset]
        cube = _only_array(_hf(repo, cube_file, data_dir), 3, cube_key)
    elif dataset in DIRECT_MAT_SCENES:
        spec = DIRECT_MAT_SCENES[dataset]
        cube = _only_array(_hf(spec["repo_id"], spec["cube_file"], data_dir), 3,
                           spec["cube_key"])
    elif dataset in URL_MAT_SCENES:
        from urllib.request import urlretrieve
        spec = URL_MAT_SCENES[dataset]
        path = data_dir / spec["cube_file"]
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            urlretrieve(spec["cube_url"], path)
        cube = _only_array(path, 3, spec["cube_key"])
    elif dataset in SCENE_FILES:
        spec = SCENE_FILES[dataset]
        cube_path = _hf(REPO_ID, spec["cube_bsq"], data_dir)
        hdr_path = _hf(REPO_ID, spec["cube_hdr"], data_dir)
        cube, _ = _read_envi_bsq(cube_path, hdr_path)
    else:
        raise ValueError(dataset)
    declared = shape_registry[dataset]
    cube = _align_known(np.asarray(cube), (declared["height"], declared["width"]))
    return cube.astype(np.float32, copy=False)


def local_scale(neighbors, q: int) -> float:
    rows = np.arange(neighbors.n)[:, None]
    nonself = np.where(neighbors.idx == rows, np.inf, neighbors.dists)
    if q > nonself.shape[1] - 1:
        raise ValueError(f"q={q} exceeds available non-self neighbours")
    qdist = np.partition(nonself, q - 1, axis=1)[:, q - 1]
    finite = qdist[np.isfinite(qdist) & (qdist > 0)]
    if not finite.size:
        raise RuntimeError("no positive finite local-scale distances")
    return float(np.median(finite))


def tau_candidates(means, dispersions, pixel_to_mode, mode_sizes, beta, thresholds, k):
    labels_by_tau, rows = [], []
    fine_k = int(np.unique(pixel_to_mode).size)
    for tau in thresholds:
        groups = reciprocal_consolidate(means, dispersions, mode_sizes, tau, k, beta)
        labels = groups[pixel_to_mode].astype(np.int32)
        stats = partition_stats(labels)
        rows.append({"prototype_threshold": float(tau), "size_exponent": float(beta),
                     "fine_clusters": fine_k,
                     "compression": float(1.0 - stats["n_clusters_full"] / fine_k), **stats})
        labels_by_tau.append(labels)
    for i, row in enumerate(rows):
        left = adjusted_rand_score(labels_by_tau[i - 1], labels_by_tau[i]) if i else None
        right = adjusted_rand_score(labels_by_tau[i], labels_by_tau[i + 1]) if i + 1 < len(rows) else None
        row["ari_previous_tau"] = None if left is None else float(left)
        row["ari_next_tau"] = None if right is None else float(right)
        row["tau_neighbor_stability"] = float(max(x for x in (left, right) if x is not None))
        row["eligible"] = bool(
            row["largest_cluster_fraction_full"] <= PROTOCOL["largest_cluster_fraction_max"]
            and row["normalized_entropy_full"] >= PROTOCOL["normalized_entropy_min"]
            and row["compression"] >= PROTOCOL["compression_min"]
            and row["tau_neighbor_stability"] >= PROTOCOL["partition_ari_min"]
        )
    # Adjacency is already enforced by tau_neighbor_stability.  The adjacent
    # partition need not itself pass compression: tau=0 is the legitimate
    # zero-compression reference for the first consolidated endpoint.
    admitted = []
    start = 0
    while start < len(rows):
        if not rows[start]["eligible"]:
            start += 1; continue
        end = start
        while end + 1 < len(rows) and rows[end + 1]["eligible"]:
            end += 1
        if end - start + 1 >= PROTOCOL["minimum_tau_run"]:
            admitted.extend(range(start, end + 1))
        start = end + 1
    if not admitted:
        return rows, None, None
    idx = min(admitted, key=lambda i: (rows[i]["n_clusters_full"],
                                       -rows[i]["tau_neighbor_stability"],
                                       rows[i]["prototype_threshold"]))
    return rows, rows[idx], labels_by_tau[idx]


def choose_across_scales(candidates: list[dict]):
    for c in candidates:
        peers = [p for p in candidates if p is not c and p["size_exponent"] == c["size_exponent"]
                 and ((p["checkpoint"] == c["checkpoint"] and abs(p["gamma_index"] - c["gamma_index"]) == 1)
                      or (p["gamma_index"] == c["gamma_index"] and abs(p["checkpoint_index"] - c["checkpoint_index"]) == 1))]
        aris = [adjusted_rand_score(c["labels"], p["labels"]) for p in peers]
        c["cross_scale_ari"] = float(max(aris)) if aris else 0.0
        c["cross_scale_admitted"] = c["cross_scale_ari"] >= PROTOCOL["cross_scale_ari_min"]
    admitted = [c for c in candidates if c["cross_scale_admitted"]]
    if not admitted:
        return None
    earliest = min(c["checkpoint"] for c in admitted)
    admitted = [c for c in admitted if c["checkpoint"] == earliest]
    size_aware = [c for c in admitted if c["size_exponent"] == 0.5]
    pool = size_aware or [c for c in admitted if c["size_exponent"] == 0.0]
    return min(pool, key=lambda c: (c["n_clusters_full"], -c["cross_scale_ari"],
                                    c["gamma"], c["prototype_threshold"]))


def select_dataset(dataset: str, data_dir: Path, output_root: Path, registry_path: Path,
                   protocol_lock: Path | None, cpu: bool):
    if dataset == HOLDOUT and (protocol_lock is None or not protocol_lock.exists()):
        raise RuntimeError("Trento selection is blocked until development_protocol_lock.json exists")
    if dataset == HOLDOUT:
        frozen = json.loads(protocol_lock.read_text())
        current_protocol_hash = hashlib.sha256(json.dumps(PROTOCOL, sort_keys=True).encode()).hexdigest()
        if frozen.get("protocol_sha256") != current_protocol_hash:
            raise RuntimeError("protocol changed after development freeze")
        if frozen.get("runner_sha256") != sha256(Path(__file__)):
            raise RuntimeError("selector source changed after development freeze")
    out = output_root / dataset
    out.mkdir(parents=True, exist_ok=True)
    lock_path = out / "selection_lock.json"
    if lock_path.exists():
        print(f"[skip] {dataset}: locked selection already exists")
        return
    registry = json.loads(registry_path.read_text())
    shape_registry = registry["datasets"]
    device = torch.device("cpu" if cpu or not torch.cuda.is_available() else "cuda")
    if device.type == "cuda": torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    cube = load_cube_only(dataset, data_dir, shape_registry)
    h, w, b = cube.shape
    flat = cube.reshape(-1, b)
    mean = flat.mean(0, dtype=np.float64).astype(np.float32)
    std = flat.std(0, dtype=np.float64).astype(np.float32); std[std < 1e-8] = 1.0
    cube = ((cube - mean) / std).astype(np.float32)
    spectra = cube.reshape(-1, b)
    graph = build_graph(cube, **PROTOCOL["graph"])
    affinity, affinity_params = compute_gaussian_affinities(
        spectra, graph.edge_i, graph.edge_j, graph.spatial_dist, seed=0)
    W, isolated = _row_normalized_weight_matrix(graph.edge_i, graph.edge_j, affinity,
                                                 graph.n_nodes, device)
    y = torch.from_numpy(spectra).to(device)
    candidates, surface_rows = [], []
    checkpoint_set = set(PROTOCOL["checkpoints"])
    checkpoint_index = {v: i for i, v in enumerate(PROTOCOL["checkpoints"])}
    for iteration in range(1, max(PROTOCOL["checkpoints"]) + 1):
        wy = torch.sparse.mm(W, y)
        ynew = 0.5 * y + 0.5 * wy
        ynew[isolated] = y[isolated]
        y = ynew
        if iteration not in checkpoint_set: continue
        if device.type == "cuda": torch.cuda.synchronize()
        cache_path = out / f"checkpoint_T{iteration}_endpoints.npz"
        if cache_path.exists():
            cached = np.load(cache_path, allow_pickle=False)
            cached_meta = json.loads(str(cached["metadata_json"].item()))
            cached_surface = json.loads(str(cached["surface_json"].item()))
            cached_labels = cached["labels"]
            for meta, lab in zip(cached_meta, cached_labels):
                candidates.append({**meta, "labels": lab.astype(np.int32, copy=False)})
            surface_rows.extend(cached_surface)
            print(f"[{dataset}] resumed cached T={iteration}; endpoint candidates={len(candidates)}", flush=True)
            continue
        yf = y.cpu().numpy()
        neighbors = compute_merge_neighbors(yf, pca_components=10, max_neighbors=50, device=device)
        s_t = local_scale(neighbors, PROTOCOL["scale_neighbor_q"])
        features = standardized_diffusion_space(yf, seed=0)
        checkpoint_candidates = []
        checkpoint_surface = []
        for gi, gamma in enumerate(PROTOCOL["gamma_grid"]):
            radius = gamma * s_t
            fine, _ = merge_from_neighbors(neighbors, radius)
            fine = majority_filter(fine, h, w, PROTOCOL["fine_merge"]["post_filter_radius"])
            means, disp, pix_to_mode, sizes, floor = mode_prototypes(features, fine)
            for beta in PROTOCOL["size_exponents"]:
                rows, endpoint, labels = tau_candidates(
                    means, disp, pix_to_mode, sizes, beta,
                    PROTOCOL["prototype_thresholds"], PROTOCOL["prototype_neighbors"])
                for row in rows:
                    record = {"dataset": dataset, "checkpoint": iteration,
                              "checkpoint_index": checkpoint_index[iteration],
                              "local_scale_s_t": s_t, "gamma": gamma,
                              "gamma_index": gi, "fine_radius": radius,
                              "dispersion_floor": floor, **row}
                    surface_rows.append(record); checkpoint_surface.append(record)
                if endpoint is not None:
                    candidate = {"checkpoint": iteration,
                                 "checkpoint_index": checkpoint_index[iteration],
                                 "local_scale_s_t": s_t, "gamma": gamma,
                                 "gamma_index": gi, "fine_radius": radius,
                                 "dispersion_floor": floor, **endpoint, "labels": labels}
                    candidates.append(candidate); checkpoint_candidates.append(candidate)
        checkpoint_meta = [{k: v for k, v in c.items() if k != "labels"}
                           for c in checkpoint_candidates]
        checkpoint_labels = np.stack([c["labels"] for c in checkpoint_candidates]) \
            if checkpoint_candidates else np.empty((0, graph.n_nodes), dtype=np.int32)
        temporary_cache = cache_path.with_suffix(".tmp.npz")
        np.savez(temporary_cache, labels=checkpoint_labels,
                 metadata_json=np.asarray(json.dumps(checkpoint_meta)),
                 surface_json=np.asarray(json.dumps(checkpoint_surface)))
        os.replace(temporary_cache, cache_path)
        del yf, neighbors, features
        if device.type == "cuda": torch.cuda.empty_cache()
        print(f"[{dataset}] completed T={iteration}; endpoint candidates={len(candidates)}", flush=True)
    chosen = choose_across_scales(candidates)
    surface_path = out / "label_free_surface.csv"
    with surface_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=surface_rows[0].keys())
        writer.writeheader(); writer.writerows(surface_rows)
    audit_candidates = [{k: v for k, v in c.items() if k != "labels"} for c in candidates]
    (out / "endpoint_candidates.json").write_text(json.dumps(audit_candidates, indent=2))
    if chosen is None:
        lock = {"status": "abstain", "dataset": dataset,
                "reason": "no endpoint passed all pre-specified label-free gates",
                "selection_uses_reference_labels": False}
    else:
        labels_path = out / "selected_labels.npy"
        np.save(labels_path, chosen["labels"].reshape(h, w))
        selected = {k: v for k, v in chosen.items() if k != "labels"}
        selector_status = ("selected_size_aware" if selected["size_exponent"] == 0.5
                           else "fallback_conservative")
        lock = {"status": "selected", "selector_status": selector_status,
                "dataset": dataset, "selected": selected,
                "selected_labels": labels_path.name, "selected_labels_sha256": sha256(labels_path),
                "selection_uses_reference_labels": False,
                "protocol_sha256": hashlib.sha256(json.dumps(PROTOCOL, sort_keys=True).encode()).hexdigest(),
                "runner_sha256": sha256(Path(__file__)),
                "development_protocol_lock_sha256": sha256(protocol_lock) if protocol_lock else None,
                "runtime_seconds": time.perf_counter() - t0,
                "device": str(device),
                "peak_gpu_memory_mb": float(torch.cuda.max_memory_allocated()/1e6) if device.type == "cuda" else None,
                "graph": {"nodes": graph.n_nodes, "directed_edges": int(graph.edge_i.size),
                          "sigma_spectral": affinity_params.sigma_spectral,
                          "sigma_spatial": affinity_params.sigma_spatial}}
    lock_path.write_text(json.dumps(lock, indent=2))
    print(json.dumps(lock, indent=2), flush=True)


def benchmark_selected(dataset: str, data_dir: Path, output_root: Path,
                       registry_path: Path, cpu: bool):
    """Measure one end-to-end run of the locked configuration without labels."""
    out = output_root / dataset
    profile_path = out / "selected_run_profile.json"
    if profile_path.exists():
        print(f"[skip] {dataset}: selected-run profile already exists")
        return
    lock = json.loads((out / "selection_lock.json").read_text())
    if lock["status"] != "selected":
        print(f"[skip] {dataset}: selector abstained")
        return
    selected = lock["selected"]
    registry = json.loads(registry_path.read_text())
    device = torch.device("cpu" if cpu or not torch.cuda.is_available() else "cuda")
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    start = time.perf_counter()
    cube = load_cube_only(dataset, data_dir, registry["datasets"])
    h, w, b = cube.shape
    flat = cube.reshape(-1, b)
    mean = flat.mean(0, dtype=np.float64).astype(np.float32)
    std = flat.std(0, dtype=np.float64).astype(np.float32)
    std[std < 1e-8] = 1.0
    cube = ((cube - mean) / std).astype(np.float32)
    spectra = cube.reshape(-1, b)
    graph = build_graph(cube, **PROTOCOL["graph"])
    affinity, _ = compute_gaussian_affinities(
        spectra, graph.edge_i, graph.edge_j, graph.spatial_dist, seed=0)
    W, isolated = _row_normalized_weight_matrix(
        graph.edge_i, graph.edge_j, affinity, graph.n_nodes, device)
    y = torch.from_numpy(spectra).to(device)
    for _ in range(int(selected["checkpoint"])):
        wy = torch.sparse.mm(W, y)
        y_new = 0.5 * y + 0.5 * wy
        y_new[isolated] = y[isolated]
        y = y_new
    if device.type == "cuda":
        torch.cuda.synchronize()
    y_final = y.cpu().numpy()
    neighbors = compute_merge_neighbors(
        y_final, pca_components=10, max_neighbors=50, device=device)
    s_t = local_scale(neighbors, PROTOCOL["scale_neighbor_q"])
    radius = float(selected["gamma"]) * s_t
    fine, _ = merge_from_neighbors(neighbors, radius)
    fine = majority_filter(
        fine, h, w, PROTOCOL["fine_merge"]["post_filter_radius"])
    features = standardized_diffusion_space(y_final, seed=0)
    means, dispersions, pixel_to_mode, mode_sizes, floor = mode_prototypes(
        features, fine)
    groups = reciprocal_consolidate(
        means, dispersions, mode_sizes,
        float(selected["prototype_threshold"]),
        PROTOCOL["prototype_neighbors"],
        float(selected["size_exponent"]),
    )
    labels = groups[pixel_to_mode].astype(np.int32).reshape(h, w)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    locked_labels = np.load(out / lock["selected_labels"])
    reproduction_ari = float(adjusted_rand_score(
        locked_labels.reshape(-1), labels.reshape(-1)))
    if reproduction_ari < 0.999:
        raise RuntimeError(
            f"selected-run reproduction ARI is only {reproduction_ari:.6f}")
    profile = {
        "dataset": dataset,
        "definition": "end-to-end selected configuration: cached/local cube load, z-score, graph, affinity, diffusion, normalized fine-radius merge, cleanup, prototype construction, and one consolidation endpoint",
        "selected_run_runtime_seconds": elapsed,
        "selected_run_peak_gpu_memory_mb": (
            float(torch.cuda.max_memory_allocated() / 1e6)
            if device.type == "cuda" else None),
        "automatic_search_runtime_seconds": lock["runtime_seconds"],
        "automatic_search_peak_gpu_memory_mb": lock["peak_gpu_memory_mb"],
        "realized_local_scale_s_t": s_t,
        "realized_fine_radius": radius,
        "dispersion_floor": floor,
        "partition_reproduction_ari": reproduction_ari,
        "reference_labels_loaded": False,
    }
    profile_path.write_text(json.dumps(profile, indent=2))
    print(json.dumps(profile, indent=2))


def freeze_development(output_root: Path, target: Path):
    records = {}
    for ds in DEVELOPMENT:
        path = output_root / ds / "selection_lock.json"
        if not path.exists(): raise RuntimeError(f"missing development lock: {path}")
        record = json.loads(path.read_text())
        if record["status"] not in ("selected", "abstain"):
            raise RuntimeError(f"invalid lock for {ds}")
        records[ds] = {"path": str(path), "sha256": sha256(path), "status": record["status"]}
    payload = {"status": "frozen_before_trento", "protocol": PROTOCOL,
               "protocol_sha256": hashlib.sha256(json.dumps(PROTOCOL, sort_keys=True).encode()).hexdigest(),
               "runner_sha256": sha256(Path(__file__)),
               "development_locks": records, "reference_labels_used_by_selector": False,
               "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2))
    print(f"Frozen development protocol: {target}\nSHA-256: {sha256(target)}")


def evaluate_dataset(dataset: str, data_dir: Path, output_root: Path):
    out = output_root / dataset
    lock_path = out / "selection_lock.json"
    lock = json.loads(lock_path.read_text())
    if lock["status"] != "selected":
        print(f"[skip evaluation] {dataset}: selector abstained"); return
    labels_path = out / lock["selected_labels"]
    if sha256(labels_path) != lock["selected_labels_sha256"]:
        raise RuntimeError("selected partition hash mismatch")
    # This is the first command in this workflow that loads the reference map.
    scene = load_scene(dataset, data_dir=data_dir)
    labels = np.load(labels_path)
    metrics = evaluate(scene.gt, labels)
    payload = {"dataset": dataset, "evaluation_after_selection_lock": True,
               "selection_lock_sha256": sha256(lock_path), **metrics}
    (out / "reference_evaluation.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


def aggregate(output_root: Path, registry_path: Path):
    registry = json.loads(registry_path.read_text())
    rows = []
    for ds in ALL_DATASETS:
        lock_path = output_root / ds / "selection_lock.json"
        if not lock_path.exists(): continue
        lock = json.loads(lock_path.read_text())
        row = {"dataset": ds,
               "selection_status": lock.get("selector_status", lock["status"])}
        if lock["status"] == "selected":
            s = lock["selected"]
            row.update({"T": s["checkpoint"], "s_T": s["local_scale_s_t"],
                        "gamma": s["gamma"], "fine_radius": s["fine_radius"],
                        "beta": s["size_exponent"], "tau": s["prototype_threshold"],
                        "Khat": s["n_clusters_full"], "largest_fraction": s["largest_cluster_fraction_full"],
                        "entropy": s["normalized_entropy_full"], "compression": s["compression"],
                        "tau_neighbor_stability": s["tau_neighbor_stability"],
                        "cross_scale_ari": s["cross_scale_ari"],
                        "acceptance_stability": min(s["tau_neighbor_stability"],
                                                    s["cross_scale_ari"]),
                        "runtime_seconds": lock["runtime_seconds"],
                        "peak_gpu_memory_mb": lock["peak_gpu_memory_mb"]})
        profile_path = output_root / ds / "selected_run_profile.json"
        if profile_path.exists():
            profile = json.loads(profile_path.read_text())
            row.update({
                "automatic_search_runtime_seconds": profile["automatic_search_runtime_seconds"],
                "automatic_search_peak_gpu_memory_mb": profile["automatic_search_peak_gpu_memory_mb"],
                "selected_run_runtime_seconds": profile["selected_run_runtime_seconds"],
                "selected_run_peak_gpu_memory_mb": profile["selected_run_peak_gpu_memory_mb"],
            })
        ep = output_root / ds / "reference_evaluation.json"
        if ep.exists():
            e = json.loads(ep.read_text())
            row.update({"OA": e["overall_accuracy"], "BA": e["balanced_accuracy"],
                        "NMI": e["nmi"], "ARI": e["ari"]})
        frozen_path = ROOT / registry["datasets"][ds]["method_source"]
        if frozen_path.exists():
            frozen = json.loads(frozen_path.read_text())
            threshold = float(frozen["label_free_selected"]["prototype_threshold"])
            curve_row = min(frozen["curve"], key=lambda x: abs(float(x["prototype_threshold"]) - threshold))
            frozen_metrics = {"OA": curve_row["overall_accuracy"],
                              "BA": curve_row["balanced_accuracy"],
                              "NMI": curve_row["nmi"], "ARI": curve_row["ari"]}
            for name, value in frozen_metrics.items():
                row[f"frozen_{name}"] = value
                if name in row: row[f"delta_{name}"] = row[name] - value
            info = registry["datasets"][ds]
            row.update({
                "earlier_T": info["fine_checkpoint"],
                "earlier_fine_radius": info["fine_radius"],
                "earlier_tau": info["prototype_threshold"],
                "earlier_beta": info["size_exponent"],
                "earlier_regions": info["n_clusters_full"],
                "earlier_runtime_seconds": frozen.get("total_runtime_seconds"),
                "earlier_peak_gpu_memory_mb": frozen.get("peak_gpu_memory_mb"),
            })
        rows.append(row)
    path = output_root / "automatic_selection_master_table.csv"
    fields = sorted({k for r in rows for k in r}, key=lambda x: (x != "dataset", x))
    with path.open("w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fields); wr.writeheader(); wr.writerows(rows)
    evaluated = [r for r in rows if "OA" in r]
    summary = {
        "automatic_selections": sum(
            r["selection_status"] in ("selected", "selected_size_aware", "fallback_conservative")
            for r in rows),
        "abstentions": sum(r["selection_status"] == "abstain" for r in rows),
        "conservative_beta0_fallbacks": sum(
            r["selection_status"] == "fallback_conservative" for r in rows),
        "evaluated_datasets": len(evaluated),
        "mean_automatic_metrics": {
            m: float(np.mean([r[m] for r in evaluated]))
            for m in ("OA", "BA", "NMI", "ARI")
        },
        "mean_earlier_metrics": {
            m: float(np.mean([r[f"frozen_{m}"] for r in evaluated
                              if f"frozen_{m}" in r]))
            for m in ("OA", "BA", "NMI", "ARI")
        },
        "mean_metric_change_vs_frozen": {
            m: float(np.mean([r[f"delta_{m}"] for r in evaluated if f"delta_{m}" in r]))
            for m in ("OA", "BA", "NMI", "ARI")
        },
        "important_holdout_note": "Trento is a locked-rule transfer audit, not a newly untouched holdout, because its earlier reference scores are already known.",
    }
    (output_root / "automatic_selection_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Saved {path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["select", "freeze-development",
                                       "benchmark-selected", "evaluate", "aggregate"])
    p.add_argument("--dataset", choices=ALL_DATASETS)
    p.add_argument("--data-dir", type=Path, default=ROOT / "data")
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--registry", type=Path,
                   default=ROOT / "results/provenance/frozen_results_registry.json")
    p.add_argument("--protocol-lock", type=Path)
    p.add_argument("--cpu", action="store_true")
    a = p.parse_args()
    if a.command in ("select", "benchmark-selected", "evaluate") and not a.dataset:
        p.error("--dataset is required")
    if a.command == "select":
        select_dataset(a.dataset, a.data_dir, a.output_root, a.registry, a.protocol_lock, a.cpu)
    elif a.command == "freeze-development":
        target = a.protocol_lock or a.output_root / "development_protocol_lock.json"
        freeze_development(a.output_root, target)
    elif a.command == "benchmark-selected":
        benchmark_selected(a.dataset, a.data_dir, a.output_root, a.registry, a.cpu)
    elif a.command == "evaluate": evaluate_dataset(a.dataset, a.data_dir, a.output_root)
    else: aggregate(a.output_root, a.registry)


if __name__ == "__main__": main()
