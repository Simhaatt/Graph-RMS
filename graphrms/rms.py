"""Graph-RMS finite graph-diffusion update.

The reported Graph-RMS procedure is inspired by repeated local weighted-mean
updates used in mean-shift mode seeking, but it is not mathematically
equivalent to conventional mean shift. The graph support and transition
weights are fixed before the reported damped diffusion, and the representation
is retained at a finite checkpoint before fine-mode formation and prototype
consolidation.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import torch
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

from .device import get_device


@dataclasses.dataclass
class RMSResult:
    y_final: np.ndarray      # (N, B) converged positions
    labels: np.ndarray       # (N,) int cluster id per pixel
    n_clusters: int
    n_iters: int
    movement_history: list[float]


def _row_normalized_weight_matrix(edge_i: np.ndarray, edge_j: np.ndarray, affinities: np.ndarray,
                                   n_nodes: int, device: torch.device):
    ei = torch.from_numpy(edge_i).long().to(device)
    ej = torch.from_numpy(edge_j).long().to(device)
    aff = torch.from_numpy(affinities).float().to(device)
    indices = torch.stack([ei, ej])
    A = torch.sparse_coo_tensor(indices, aff, size=(n_nodes, n_nodes)).coalesce()

    row_sums = torch.sparse.sum(A, dim=1).to_dense()
    isolated = row_sums <= 1e-12
    inv = torch.zeros_like(row_sums)
    inv[~isolated] = 1.0 / row_sums[~isolated]

    vals = A.values() * inv[A.indices()[0]]
    W = torch.sparse_coo_tensor(A.indices(), vals, size=(n_nodes, n_nodes)).coalesce()
    # CSR has a dedicated cuSPARSE SpMM kernel; COO sparse-dense matmul on GPU
    # falls back to a much slower path (this was the actual cause of graph-RMS
    # taking ~0.8s/iteration instead of the ~0.02-0.05s it should on an A100).
    W = W.to_sparse_csr()
    return W, isolated


def run_graph_rms(y0: np.ndarray, edge_i: np.ndarray, edge_j: np.ndarray,
                   affinities: np.ndarray, n_nodes: int, max_iter: int = 50,
                   tol: float = 1e-4, alpha: float = 0.5,
                   anchor_strength: float = 0.0,
                   device: torch.device | None = None) -> tuple[np.ndarray, int, list[float]]:
    """alpha < 1 makes this a "lazy" update, y <- (1-alpha)*y + alpha*(W@y),
    instead of a full replacement. The graph has no self-loops (an edge
    (i, i) was explicitly excluded in graph.py), so a full replacement can
    2-cycle forever on small reciprocal neighbourhoods (y_i <-> y_j swapping
    every step) instead of converging -- damping guarantees convergence the
    same way a lazy random walk does."""
    if not 0.0 <= anchor_strength < 1.0:
        raise ValueError("anchor_strength must be in [0, 1)")
    device = device or get_device()
    W, isolated = _row_normalized_weight_matrix(edge_i, edge_j, affinities, n_nodes, device)
    x0 = torch.from_numpy(y0.astype(np.float32)).to(device)
    y = x0.clone()

    history = []
    n_iters = 0
    for it in range(max_iter):
        wy = torch.sparse.mm(W, y)
        # Restart/anchoring retains a controlled amount of the original
        # spectrum at every step, limiting long-run consensus collapse. Setting
        # anchor_strength=0 exactly reproduces the original lazy diffusion.
        proposal = anchor_strength * x0 + (1 - anchor_strength) * wy
        y_new = (1 - alpha) * y + alpha * proposal
        y_new[isolated] = y[isolated]  # isolated pixels don't move
        movement = torch.sqrt(((y_new - y) ** 2).sum(dim=1)).max().item()
        history.append(movement)
        y = y_new
        n_iters = it + 1
        if movement < tol:
            break
    return y.cpu().numpy(), n_iters, history


def run_dynamic_graph_rms(spectra: np.ndarray, edge_i: np.ndarray, edge_j: np.ndarray,
                           spatial_dist: np.ndarray, sigma_spectral: float, sigma_spatial: float,
                           n_nodes: int, max_iter: int = 100, tol: float = 1e-4, alpha: float = 0.5,
                           blurring: bool = True, use_spatial: bool = True,
                           device: torch.device | None = None) -> tuple[np.ndarray, int, list[float]]:
    """Genuine graph mean-shift: the spectral affinity is RECOMPUTED from the
    current positions at every iteration (unlike run_graph_rms, which applies a
    fixed diffusion matrix). The spatial candidate graph (MNN edges) stays fixed;
    only the edge weights adapt.

      blurring=True  : weights and the aggregated neighbours both use the current
                       positions y^(t) (blurring mean-shift; collapses to consensus
                       in the limit, clusters come from the transient).
      blurring=False : weights use current y_i vs. the FIXED original neighbour
                       spectra x_j and aggregate the original x_j (non-blurring
                       mean-shift; seeks genuine density modes without collapse).

    The spatial term P_ij = exp(-d_spatial^2 / 2 sigma_p^2) is fixed either way.
    """
    device = device or get_device()
    x0 = torch.from_numpy(spectra.astype(np.float32)).to(device)
    ei = torch.from_numpy(edge_i).long().to(device)
    ej = torch.from_numpy(edge_j).long().to(device)
    b = x0.shape[1]
    if use_spatial:
        sp2 = (torch.from_numpy(spatial_dist).float().to(device)) ** 2
        pspat = torch.exp(-sp2 / (2 * sigma_spatial ** 2))
    else:
        pspat = torch.ones(ei.shape[0], device=device)

    y = x0.clone()
    history, n_iters = [], 0
    for it in range(max_iter):
        nbr = y if blurring else x0                      # what we aggregate / compare against
        d2 = ((y[ei] - nbr[ej]) ** 2).sum(dim=1)         # current position -> neighbour
        w = torch.exp(-d2 / (2 * sigma_spectral ** 2)) * pspat
        num = torch.zeros(n_nodes, b, device=device)
        num.index_add_(0, ei, w.unsqueeze(-1) * nbr[ej])
        den = torch.zeros(n_nodes, device=device)
        den.index_add_(0, ei, w)
        valid = den > 1e-12
        y_new = y.clone()
        y_new[valid] = (1 - alpha) * y[valid] + alpha * (num[valid] / den[valid].unsqueeze(-1))
        movement = torch.sqrt(((y_new - y) ** 2).sum(dim=1)).max().item()
        history.append(movement)
        y = y_new
        n_iters = it + 1
        if movement < tol:
            break
    return y.cpu().numpy(), n_iters, history


@dataclasses.dataclass
class MergeNeighbors:
    """Precomputed k-nearest-neighbour distances in the standardized merge
    space, reusable across many tol values without re-querying."""
    dists: np.ndarray  # (N, k)
    idx: np.ndarray    # (N, k)
    n: int


def _gpu_knn(x: np.ndarray, k: int, device: torch.device, tile: int = 2048):
    """Batched exact k-NN on GPU via tiled ||a-b||^2 = |a|^2+|b|^2-2 a.b.
    For the merge step on a full scene (e.g. 446k points) this turns a
    multi-minute sklearn CPU query into seconds on an H100."""
    xt = torch.from_numpy(x).float().to(device)
    sq = (xt * xt).sum(1)                      # (n,)
    n = xt.shape[0]
    dists = torch.empty(n, k, device=device)
    idx = torch.empty(n, k, dtype=torch.long, device=device)
    for s in range(0, n, tile):
        e = min(s + tile, n)
        # rank with the fast expansion (exact ordering; slight value error near 0)
        d2 = sq[s:e, None] + sq[None, :] - 2.0 * (xt[s:e] @ xt.T)
        d2.clamp_(min=0)
        _, ti = torch.topk(d2, k, dim=1, largest=False)
        # recompute the k selected distances exactly by direct subtraction
        diff = xt[s:e, None, :] - xt[ti]              # (tile, k, dim)
        dists[s:e] = torch.sqrt((diff * diff).sum(-1))
        idx[s:e] = ti
    return dists.cpu().numpy(), idx.cpu().numpy()


def compute_merge_neighbors(y_final: np.ndarray, pca_components: int = 10,
                             max_neighbors: int = 50,
                             device: torch.device | None = None) -> MergeNeighbors:
    """The expensive part of merging: fit PCA + a k-NN index once. Uses a
    fixed-k query (cost ~ O(N*max_neighbors)) rather than sklearn's
    radius_neighbors_graph, whose cost blows up once a radius approaches/
    exceeds the natural neighbour scale -- at that point almost every point
    is "within radius" of almost every other, and the query can hang or
    exhaust memory on a few hundred thousand points.

    Runs the k-NN on GPU when one is available (auto-detected), which is the
    difference between minutes and seconds on a full-scene merge."""
    n_comp = min(pca_components, y_final.shape[1], y_final.shape[0] - 1)
    reduced = PCA(n_components=n_comp, random_state=0).fit_transform(y_final)
    mean, std = reduced.mean(axis=0), reduced.std(axis=0)
    std[std < 1e-8] = 1.0
    standardized = ((reduced - mean) / std).astype(np.float32)

    n = standardized.shape[0]
    k = min(max_neighbors, n - 1)
    device = device or get_device()
    if device.type == "cuda":
        dists, idx = _gpu_knn(standardized, k, device)
    else:
        nn = NearestNeighbors(n_neighbors=k).fit(standardized)
        dists, idx = nn.kneighbors(standardized)
    return MergeNeighbors(dists=dists, idx=idx, n=n)


def merge_from_neighbors(neighbors: MergeNeighbors, tol: float) -> tuple[np.ndarray, int]:
    """Cheap part: threshold precomputed k-NN distances at `tol` and take
    connected components. Safe to call many times (e.g. a tol sweep) against
    one compute_merge_neighbors() result -- no re-querying.

    Tradeoff of the fixed-k neighbour cap: if two points that should merge
    aren't within each other's max_neighbors closest candidates, they won't
    merge even if within tol -- rerun compute_merge_neighbors with a larger
    max_neighbors if that's a concern."""
    n, k = neighbors.dists.shape
    rows = np.repeat(np.arange(n, dtype=np.int64), k)
    cols = neighbors.idx.reshape(-1)
    within_tol = neighbors.dists.reshape(-1) <= tol
    rows, cols = rows[within_tol], cols[within_tol]

    graph = sparse.coo_matrix((np.ones(rows.shape[0], dtype=np.uint8), (rows, cols)), shape=(n, n))
    n_clusters, labels = connected_components(graph, directed=False)
    return labels.astype(np.int32), n_clusters


def merge_modes(y_final: np.ndarray, tol: float = 0.15, pca_components: int = 10,
                 max_neighbors: int = 50) -> tuple[np.ndarray, int]:
    """Convenience one-shot wrapper around compute_merge_neighbors +
    merge_from_neighbors for a single tol value. For a tol sweep, call those
    two directly so the expensive PCA/k-NN fit only happens once."""
    neighbors = compute_merge_neighbors(y_final, pca_components=pca_components, max_neighbors=max_neighbors)
    return merge_from_neighbors(neighbors, tol=tol)


def graph_rms_cluster(spectra: np.ndarray, edge_i: np.ndarray, edge_j: np.ndarray,
                       affinities: np.ndarray, n_nodes: int, max_iter: int = 50,
                       tol: float = 1e-4, alpha: float = 0.5, merge_tol: float = 0.15,
                       merge_pca_components: int = 10,
                       anchor_strength: float = 0.0,
                       device: torch.device | None = None) -> RMSResult:
    y_final, n_iters, history = run_graph_rms(spectra, edge_i, edge_j, affinities, n_nodes,
                                               max_iter=max_iter, tol=tol, alpha=alpha,
                                               anchor_strength=anchor_strength, device=device)
    labels, n_clusters = merge_modes(y_final, tol=merge_tol, pca_components=merge_pca_components)
    return RMSResult(y_final=y_final, labels=labels, n_clusters=n_clusters,
                      n_iters=n_iters, movement_history=history)
