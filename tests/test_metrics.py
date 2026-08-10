from __future__ import annotations

import numpy as np

from graphrms.metrics import evaluate


def test_metrics_are_permutation_invariant_for_perfect_partition():
    truth = np.array([[1, 1, 2, 2], [1, 1, 2, 2]])
    prediction = np.array([[9, 9, 4, 4], [9, 9, 4, 4]])
    metrics = evaluate(truth, prediction)
    assert metrics["overall_accuracy"] == 1.0
    assert metrics["balanced_accuracy"] == 1.0
    assert metrics["nmi"] == 1.0
    assert metrics["ari"] == 1.0


def test_zero_reference_pixels_are_excluded():
    truth = np.array([0, 1, 1, 2, 2])
    prediction = np.array([99, 4, 4, 8, 8])
    metrics = evaluate(truth, prediction)
    assert metrics["overall_accuracy"] == 1.0

