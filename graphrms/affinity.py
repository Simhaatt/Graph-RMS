"""Fixed Gaussian spectral-spatial affinity used by Graph-RMS.

The affinity is a deterministic function of supported-edge spectral and
spatial distances. Kernel widths are estimated with a seeded median heuristic
over at most 50,000 stored supported edges and are floored at ``1e-6``.
"""

from __future__ import annotations

import dataclasses

import numpy as np


@dataclasses.dataclass
class GaussianAffinityParams:
    sigma_spectral: float
    sigma_spatial: float


def median_heuristic_sigmas(spectra: np.ndarray, edge_i: np.ndarray, edge_j: np.ndarray,
                             spatial_dist: np.ndarray, sample_size: int = 50_000,
                             seed: int = 0) -> tuple[float, float]:
    """Median pairwise distance over a sample of candidate edges -- the
    standard, data-driven way to pick a Gaussian kernel width."""
    e = edge_i.shape[0]
    rng = np.random.default_rng(seed)
    sample = rng.choice(e, size=min(sample_size, e), replace=False)
    sample_spectral_dist = np.sqrt(((spectra[edge_i[sample]] - spectra[edge_j[sample]]) ** 2).sum(axis=1))
    sigma_spectral = max(float(np.median(sample_spectral_dist)), 1e-6)
    sigma_spatial = max(float(np.median(spatial_dist[sample])), 1e-6)
    return sigma_spectral, sigma_spatial


def compute_gaussian_affinities(spectra: np.ndarray, edge_i: np.ndarray, edge_j: np.ndarray,
                                 spatial_dist: np.ndarray, sigma_spectral: float | None = None,
                                 sigma_spatial: float | None = None, batch_edges: int = 500_000,
                                 sample_size: int = 50_000, use_spatial: bool = True,
                                 seed: int = 0) -> tuple[np.ndarray, GaussianAffinityParams]:
    e = edge_i.shape[0]

    if sigma_spectral is None or sigma_spatial is None:
        ms, mp = median_heuristic_sigmas(spectra, edge_i, edge_j, spatial_dist, sample_size, seed)
        if sigma_spectral is None:
            sigma_spectral = ms
        if sigma_spatial is None:
            sigma_spatial = mp

    out = np.empty(e, dtype=np.float32)
    for start in range(0, e, batch_edges):
        end = min(start + batch_edges, e)
        d = np.sqrt(((spectra[edge_i[start:end]] - spectra[edge_j[start:end]]) ** 2).sum(axis=1))
        s = np.exp(-(d ** 2) / (2 * sigma_spectral ** 2))
        if use_spatial:  # P_ij term; off = spectral-only affinity (spatial ablation)
            p = np.exp(-(spatial_dist[start:end] ** 2) / (2 * sigma_spatial ** 2))
            out[start:end] = (s * p).astype(np.float32)
        else:
            out[start:end] = s.astype(np.float32)

    return out, GaussianAffinityParams(sigma_spectral=sigma_spectral, sigma_spatial=sigma_spatial)
