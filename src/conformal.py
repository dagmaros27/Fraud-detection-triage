"""Conformal prediction utilities for fraud triage routing."""

from __future__ import annotations

import math

import numpy as np


def _stack_binary_probs(probs: np.ndarray) -> np.ndarray:
    """Convert fraud probabilities into two-class probability vectors."""

    probs_array = np.asarray(probs, dtype=float)
    return np.column_stack([1.0 - probs_array, probs_array])


def compute_nonconformity_scores(probs_cal, y_cal) -> np.ndarray:
    """Compute APS nonconformity scores on the calibration set.

    Parameters
    ----------
    probs_cal:
        Fraud-class probabilities for calibration samples.
    y_cal:
        Ground-truth binary labels for calibration samples.

    Returns
    -------
    np.ndarray
        APS nonconformity scores with shape ``(n_cal,)``.
    """

    class_probs = _stack_binary_probs(probs_cal)
    y_array = np.asarray(y_cal, dtype=int)
    scores = np.zeros(len(y_array), dtype=float)

    for index, (sample_probs, true_label) in enumerate(zip(class_probs, y_array)):
        sorted_indices = np.argsort(sample_probs)[::-1]
        cumulative_mass = np.cumsum(sample_probs[sorted_indices])
        true_rank = int(np.where(sorted_indices == true_label)[0][0])
        scores[index] = cumulative_mass[true_rank]

    return scores


def compute_threshold(nonconformity_scores, alpha: float = 0.1) -> float:
    """Compute the APS conformal threshold using the finite-sample quantile.

    Parameters
    ----------
    nonconformity_scores:
        Calibration nonconformity scores.
    alpha:
        Miscoverage level.

    Returns
    -------
    float
        The conformal threshold ``tau_hat``.
    """

    scores = np.sort(np.asarray(nonconformity_scores, dtype=float))
    n = len(scores)
    if n == 0:
        raise ValueError("nonconformity_scores must be non-empty.")

    quantile_rank = math.ceil((1.0 - alpha) * (n + 1))
    quantile_rank = min(max(quantile_rank, 1), n)
    return float(scores[quantile_rank - 1])


def get_prediction_sets(probs, tau: float) -> np.ndarray:
    """Compute APS prediction-set sizes for binary probabilities.

    Parameters
    ----------
    probs:
        Fraud-class probabilities.
    tau:
        Conformal threshold.

    Returns
    -------
    np.ndarray
        Prediction-set sizes with values 1 or 2.
    """

    class_probs = _stack_binary_probs(probs)
    set_sizes = np.zeros(len(class_probs), dtype=int)

    for index, sample_probs in enumerate(class_probs):
        sorted_probs = np.sort(sample_probs)[::-1]
        cumulative_mass = np.cumsum(sorted_probs)
        first_exceeding = np.searchsorted(cumulative_mass, tau, side="left")
        set_sizes[index] = min(first_exceeding + 1, len(sorted_probs))

    return set_sizes


def get_routing_signal(probs, tau: float) -> np.ndarray:
    """Map prediction-set sizes to routing decisions.

    Parameters
    ----------
    probs:
        Fraud-class probabilities.
    tau:
        Conformal threshold.

    Returns
    -------
    np.ndarray
        Array containing ``"auto"`` or ``"escalate"`` for each sample.
    """

    set_sizes = get_prediction_sets(probs, tau)
    return np.where(set_sizes == 1, "auto", "escalate")


def get_three_zone_routing(
    probs,
    tau: float,
    low_thresh: float = 0.05,
    high_thresh: float = 0.80,
) -> np.ndarray:
    """Construct a deployment-style three-zone routing signal.

    Parameters
    ----------
    probs:
        Fraud-class probabilities.
    tau:
        Conformal threshold used to compute prediction-set sizes.
    low_thresh:
        Lower hard threshold for clear legitimate cases.
    high_thresh:
        Upper hard threshold for clear fraud cases.

    Returns
    -------
    np.ndarray
        Array of routing labels: ``auto_approve``, ``auto_block``,
        ``auto_decide``, or ``escalate``.
    """

    probs_array = np.asarray(probs)
    set_sizes = get_prediction_sets(probs_array, tau)

    routing = np.full(len(probs_array), "escalate", dtype=object)
    routing[probs_array < low_thresh] = "auto_approve"
    routing[probs_array > high_thresh] = "auto_block"

    middle = (probs_array >= low_thresh) & (probs_array <= high_thresh)
    routing[middle & (set_sizes == 1)] = "auto_decide"
    return routing


def coverage_check(probs_test, y_test, tau: float) -> float:
    """Compute empirical coverage of APS prediction sets on the test set.

    Parameters
    ----------
    probs_test:
        Fraud-class probabilities for test samples.
    y_test:
        Ground-truth binary test labels.
    tau:
        Conformal threshold.

    Returns
    -------
    float
        Fraction of samples whose true label is included in the prediction set.
    """

    class_probs = _stack_binary_probs(probs_test)
    y_array = np.asarray(y_test, dtype=int)
    covered = np.zeros(len(y_array), dtype=bool)

    for index, (sample_probs, true_label) in enumerate(zip(class_probs, y_array)):
        sorted_indices = np.argsort(sample_probs)[::-1]
        cumulative_mass = np.cumsum(sample_probs[sorted_indices])
        first_exceeding = np.searchsorted(cumulative_mass, tau, side="left")
        set_indices = sorted_indices[: min(first_exceeding + 1, len(sorted_indices))]
        covered[index] = true_label in set_indices

    return float(covered.mean())
