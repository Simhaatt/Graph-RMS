"""Fixed, hand-picked Gaussian-kernel affinity -- the paper's original Eq. 1
(A_ij = M_ij * S_ij * P_ij with S/P Gaussian kernels) as a non-learned
alternative to the KAN affinity in kan.py/train.py.

No training step: this is a direct function of the graph's edge features.
Exists for two reasons: (1) a fast, always-available baseline when the KAN
affinity needs more tuning than time allows, (2) the ablation comparison a
paper needs anyway -- Graph-RMS+Gaussian vs. Graph-RMS+KAN -- to show the
learned affinity is actually earning its complexity rather than just being
different.

Kernel widths (sigma) are picked via the median heuristic (the standard,
principled way to set a Gaussian kernel width from data instead of guessing
a constant) unless overridden.
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
    standard, data-driven way to pick a Gaussian kernel width. Shared by the
    fixed-Gaussian and learned-metric affinities so both use identical widths
    (only the per-band weighting differs), keeping the ablation apples-to-apples."""
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
