"""Spectral-spatial mutual-nearest-neighbour (MNN) candidate graph.

Candidate neighbours for pixel i are drawn from a local spatial window
(so spatial context is baked into which edges can exist at all, not just
into a soft weight), ranked by distance in a PCA-reduced spectral space.
The MNN gate keeps an edge (i, j) only when each pixel is in the other's
candidate list. This module determines graph support; the fixed Gaussian
spectral-spatial affinity then weights the surviving edges.
"""

from __future__ import annotations

import dataclasses

import numpy as np
from sklearn.decomposition import PCA


@dataclasses.dataclass
class Graph:
    edge_i: np.ndarray       # (E,) int64, source pixel flat index, ascending/sorted
    edge_j: np.ndarray       # (E,) int64, target pixel flat index
    spatial_dist: np.ndarray  # (E,) float32, pixel-grid Euclidean distance
    indptr: np.ndarray       # (n_nodes+1,) int64, edges for node i are [indptr[i]:indptr[i+1]]
    n_nodes: int
    height: int
    width: int


def compute_pca(cube: np.ndarray, n_components: int = 30) -> np.ndarray:
    """PCA-reduce the spectral dimension, used only for neighbour search."""
    h, w, b = cube.shape
    flat = cube.reshape(-1, b)
    n_components = min(n_components, b, flat.shape[0])
    pca = PCA(n_components=n_components, random_state=0)
    reduced = pca.fit_transform(flat)
    return reduced.reshape(h, w, n_components).astype(np.float32)


def _windowed_candidates(pca_cube: np.ndarray, k: int, window_radius: int):
    """For every pixel, rank the pixels in its (2r+1)x(2r+1) spatial window
    by PCA-space distance and keep the k closest. Returns:
      nbr_idx:  (H*W, k) int64 flat index of each candidate
      nbr_dist: (H*W, k) float32 squared PCA distance to that candidate
    """
    h, w, p = pca_cube.shape
    if window_radius < 1:
        raise ValueError("window_radius must be at least 1")
    if k < 1:
        raise ValueError("k must be at least 1")
    r = window_radius
    # Do not edge-pad candidate identities. Edge padding repeats the corner or
    # border pixel for every out-of-image offset, producing several zero-distance
    # copies of self. Masking only the central offset then lets those copies fill
    # the top-k list and can leave corners isolated after the reciprocal gate.
    padded = np.pad(pca_cube, ((r, r), (r, r), (0, 0)), mode="constant")
    idx_orig = np.arange(h * w, dtype=np.int64).reshape(h, w)
    padded_idx = np.pad(idx_orig, ((r, r), (r, r)), mode="constant", constant_values=-1)

    window = 2 * r + 1
    n_offsets = window * window
    all_dists = np.empty((h, w, n_offsets), dtype=np.float32)
    all_idx = np.empty((h, w, n_offsets), dtype=np.int64)

    o = 0
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            shifted_pca = padded[r + dy: r + dy + h, r + dx: r + dx + w, :]
            shifted_idx = padded_idx[r + dy: r + dy + h, r + dx: r + dx + w]
            diff = pca_cube - shifted_pca
            all_dists[:, :, o] = np.einsum("hwp,hwp->hw", diff, diff)
            all_idx[:, :, o] = shifted_idx
            invalid = (shifted_idx < 0) | (shifted_idx == idx_orig)
            all_dists[:, :, o][invalid] = np.inf
            o += 1

    flat_dists = all_dists.reshape(h * w, n_offsets)
    flat_idx = all_idx.reshape(h * w, n_offsets)

    # A corner has only (r+1)^2-1 real neighbours. A uniform k no larger than
    # that bound guarantees every returned entry is valid for every pixel.
    k = min(k, (r + 1) ** 2 - 1)
    part = np.argpartition(flat_dists, k - 1, axis=1)[:, :k]
    row = np.arange(h * w)[:, None]
    nbr_dist = flat_dists[row, part]
    nbr_idx = flat_idx[row, part]
    order = np.argsort(nbr_dist, axis=1)
    nbr_dist = np.take_along_axis(nbr_dist, order, axis=1)
    nbr_idx = np.take_along_axis(nbr_idx, order, axis=1)
    if np.any(nbr_idx < 0) or np.any(~np.isfinite(nbr_dist)):
        raise RuntimeError("candidate construction returned an invalid neighbour")
    return nbr_idx, nbr_dist


def _repair_min_degree(mutual_graph, candidate_graph, nbr_idx: np.ndarray, min_degree: int):
    """Supplement low-degree MNN nodes with their closest candidate edges.

    The repair is deliberately opt-in: ``min_degree=0`` is the exact MNN graph.
    Added edges are symmetrized, remain inside the original spatial window, and
    are drawn in ranked spectral-distance order from the candidate list.
    """
    from scipy import sparse

    if min_degree <= 0:
        return mutual_graph
    if min_degree > nbr_idx.shape[1]:
        raise ValueError("min_degree cannot exceed the effective candidate count")

    graph = mutual_graph.tocsr()
    degrees = np.diff(graph.indptr)
    low_nodes = np.flatnonzero(degrees < min_degree)
    add_i: list[int] = []
    add_j: list[int] = []
    for i in low_nodes:
        existing = set(graph.indices[graph.indptr[i]:graph.indptr[i + 1]].tolist())
        need = min_degree - len(existing)
        for j in nbr_idx[i]:
            j = int(j)
            if j not in existing:
                add_i.append(int(i))
                add_j.append(j)
                existing.add(j)
                need -= 1
                if need == 0:
                    break

    if add_i:
        additions = sparse.coo_matrix(
            (np.ones(len(add_i), dtype=np.uint8), (add_i, add_j)),
            shape=graph.shape,
        ).tocsr()
        graph = ((graph + additions + additions.T) > 0).tocsr()
    return graph


def build_graph(cube: np.ndarray, pca_components: int = 30, k: int = 20,
                 window_radius: int = 7, mutual: bool = True,
                 min_degree: int = 0) -> Graph:
    """mutual=True keeps only reciprocal edges (the MNN gate, Eq. 1's M_ij);
    mutual=False keeps the symmetrized union of candidate edges instead -- the
    'MNN off' ablation, to show the gate is doing real work."""
    h, w, _ = cube.shape
    n = h * w
    pca_cube = compute_pca(cube, n_components=pca_components)
    nbr_idx, _ = _windowed_candidates(pca_cube, k=k, window_radius=window_radius)

    rows = np.repeat(np.arange(n, dtype=np.int64), nbr_idx.shape[1])
    cols = nbr_idx.reshape(-1)

    from scipy import sparse
    cand = sparse.coo_matrix((np.ones(rows.shape[0], dtype=np.uint8), (rows, cols)),
                              shape=(n, n)).tocsr()
    cand.data[:] = 1
    cand.sum_duplicates()
    if mutual:
        combined = cand.multiply(cand.T)                # reciprocal edges only (MNN gate)
        combined = _repair_min_degree(combined, cand, nbr_idx, min_degree)
    else:
        combined = ((cand + cand.T) > 0)                 # symmetrized union (gate off)

    combined = combined.tocoo()

    edge_i = combined.row.astype(np.int64)
    edge_j = combined.col.astype(np.int64)
    keep = edge_i != edge_j  # drop any self-loops introduced by border replication
    edge_i, edge_j = edge_i[keep], edge_j[keep]
    order = np.lexsort((edge_j, edge_i))
    edge_i, edge_j = edge_i[order], edge_j[order]

    yi, xi = edge_i // w, edge_i % w
    yj, xj = edge_j // w, edge_j % w
    spatial_dist = np.sqrt((yi - yj) ** 2 + (xi - xj) ** 2).astype(np.float32)

    # edge_i is ascending because it comes from a CSR sparse-matrix product,
    # so edges for node i form a contiguous slice; indptr marks the slices.
    assert np.all(np.diff(edge_i) >= 0), "expected edge_i sorted ascending"
    indptr = np.searchsorted(edge_i, np.arange(n + 1, dtype=np.int64))

    return Graph(edge_i=edge_i, edge_j=edge_j, spatial_dist=spatial_dist,
                 indptr=indptr.astype(np.int64), n_nodes=n, height=h, width=w)
