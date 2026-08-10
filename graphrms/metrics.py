"""Unsupervised clustering evaluated against ground truth via Hungarian
matching -- the method never sees labels during training/clustering, but
WHU-Hi-HongHu ships dense ground truth so this is the only way to confirm
the pipeline actually separates land-cover classes rather than just running.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import (adjusted_mutual_info_score, adjusted_rand_score,
                             balanced_accuracy_score, cohen_kappa_score,
                             completeness_score, homogeneity_score,
                             normalized_mutual_info_score, v_measure_score)


def hungarian_match(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, dict[int, int]]:
    """Best 1-1 mapping from predicted cluster ids to true class ids that
    maximizes overlap (via the Hungarian algorithm on a contingency table).
    Predicted clusters with no assigned true class map to -1 (always wrong)."""
    true_classes = np.unique(y_true)
    pred_classes = np.unique(y_pred)
    true_idx = {c: i for i, c in enumerate(true_classes)}
    pred_idx = {c: i for i, c in enumerate(pred_classes)}

    cost = np.zeros((len(true_classes), len(pred_classes)), dtype=np.int64)
    np.add.at(cost, (np.array([true_idx[t] for t in y_true]), np.array([pred_idx[p] for p in y_pred])), 1)

    row_ind, col_ind = linear_sum_assignment(-cost)
    mapping = {int(pred_classes[c]): int(true_classes[r]) for r, c in zip(row_ind, col_ind)}

    mapped_pred = np.full(y_pred.shape, -1, dtype=np.int64)
    for p, t in mapping.items():
        mapped_pred[y_pred == p] = t
    return mapped_pred, mapping


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, ignore_label: int = 0) -> dict:
    """Evaluate predicted cluster labels against ground truth, ignoring
    unlabeled/background pixels (ignore_label, 0 by WHU-Hi convention)."""
    mask = y_true != ignore_label
    yt, yp = y_true[mask], y_pred[mask]

    mapped_pred, mapping = hungarian_match(yt, yp)
    oa = float((mapped_pred == yt).mean())
    kappa = float(cohen_kappa_score(yt, mapped_pred))
    nmi = float(normalized_mutual_info_score(yt, yp))
    ari = float(adjusted_rand_score(yt, yp))

    n_true = int(len(np.unique(yt)))
    n_pred = int(len(np.unique(yp)))
    # Over-segmentation diagnostics: predicted cluster count vs. true classes,
    # and the fraction of labelled pixels sitting in "tiny" clusters (< 0.5% of
    # labelled pixels each) -- a direct measure of fragmentation.
    _, counts = np.unique(yp, return_counts=True)
    tiny = counts[counts < 0.005 * yp.size].sum()

    return {
        "overall_accuracy": oa,
        "balanced_accuracy": float(balanced_accuracy_score(yt, mapped_pred)),
        "kappa": kappa,
        "nmi": nmi,
        "ari": ari,
        "ami": float(adjusted_mutual_info_score(yt, yp)),
        "homogeneity": float(homogeneity_score(yt, yp)),
        "completeness": float(completeness_score(yt, yp)),
        "v_measure": float(v_measure_score(yt, yp)),
        "n_true_classes": n_true,
        "n_pred_clusters": n_pred,
        "k_hat_minus_k": int(n_pred - n_true),
        "tiny_cluster_pixel_fraction": float(tiny / max(yp.size, 1)),
        "n_labeled_pixels": int(mask.sum()),
        "cluster_to_class_mapping": mapping,
    }
