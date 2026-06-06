"""Model training and calibration utilities for fraud triage."""

from __future__ import annotations

from typing import Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit
from sklearn.metrics import roc_auc_score


def train_lgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> lgb.LGBMClassifier:
    """Train a LightGBM fraud classifier with early stopping.

    Parameters
    ----------
    X_train:
        Training features.
    y_train:
        Training labels.
    X_val:
        Validation features used for early stopping.
    y_val:
        Validation labels used for early stopping.

    Returns
    -------
    lightgbm.LGBMClassifier
        The fitted LightGBM model.
    """

    positives = int((y_train == 1).sum())
    if positives == 0:
        raise ValueError("y_train must contain at least one positive example.")

    scale_pos_weight = (y_train == 0).sum() / positives
    model = lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=64,
        min_child_samples=20,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        verbose=-1,
    )
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="auc",
        callbacks=[
            lgb.early_stopping(
                stopping_rounds=50,
                verbose=True,
                first_metric_only=True,
            ),
            lgb.log_evaluation(period=50),
        ],
    )
    print(f"Best iteration: {model.best_iteration_}")
    print(f"Best val AUC: {model.best_score_['valid_0']['auc']:.4f}")
    return model


def get_probs(model: lgb.LGBMClassifier, X: pd.DataFrame) -> np.ndarray:
    """Return positive-class predicted probabilities for a feature matrix."""

    return model.predict_proba(X)[:, 1]


def compute_ece(y_true: pd.Series | np.ndarray, probs: np.ndarray, n_bins: int = 10) -> float:
    """Compute Expected Calibration Error using equal-width confidence bins.

    Parameters
    ----------
    y_true:
        Ground-truth labels.
    probs:
        Predicted positive-class probabilities.
    n_bins:
        Number of equal-width bins across ``[0, 1]``.

    Returns
    -------
    float
        The expected calibration error.
    """

    y_true_array = np.asarray(y_true)
    probs_array = np.asarray(probs)
    if len(probs_array) == 0:
        return 0.0

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(probs_array, bin_edges[1:-1], right=True)
    n_samples = len(probs_array)
    ece = 0.0

    for bin_index in range(n_bins):
        mask = bin_ids == bin_index
        if not np.any(mask):
            continue

        bin_accuracy = y_true_array[mask].mean()
        bin_confidence = probs_array[mask].mean()
        bin_weight = mask.sum() / n_samples
        ece += bin_weight * abs(bin_accuracy - bin_confidence)

    return float(ece)


def temperature_scale(
    probs_val: np.ndarray,
    y_val: pd.Series | np.ndarray,
    probs_to_scale: np.ndarray,
) -> Tuple[np.ndarray, float]:
    """Fit a temperature on validation probabilities and scale another set.

    Parameters
    ----------
    probs_val:
        Validation-set probabilities used to fit the temperature.
    y_val:
        Validation labels.
    probs_to_scale:
        Probabilities to transform using the learned temperature.

    Returns
    -------
    tuple[np.ndarray, float]
        The scaled probabilities and the learned temperature value.
    """

    epsilon = 1e-7
    probs_val_clipped = np.clip(np.asarray(probs_val), epsilon, 1.0 - epsilon)
    probs_to_scale_clipped = np.clip(np.asarray(probs_to_scale), epsilon, 1.0 - epsilon)
    y_val_array = np.asarray(y_val)
    val_logits = logit(probs_val_clipped)
    target_logits = logit(probs_to_scale_clipped)

    def nll_objective(temperature: float) -> float:
        scaled = expit(val_logits / temperature)
        scaled = np.clip(scaled, epsilon, 1.0 - epsilon)
        return float(
            -np.mean(
                y_val_array * np.log(scaled) + (1.0 - y_val_array) * np.log(1.0 - scaled)
            )
        )

    result = minimize_scalar(nll_objective, bounds=(0.1, 10.0), method="bounded")
    temperature = float(result.x)
    scaled_probs = expit(target_logits / temperature)
    return scaled_probs, temperature


def reliability_diagram_data(
    y_true: pd.Series | np.ndarray,
    probs: np.ndarray,
    n_bins: int = 10,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Prepare equal-width binned calibration data for reliability plotting.

    Parameters
    ----------
    y_true:
        Ground-truth labels.
    probs:
        Predicted positive-class probabilities.
    n_bins:
        Number of equal-width bins across ``[0, 1]``.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        Arrays containing bin centers, accuracies, confidences, and sizes.
    """

    y_true_array = np.asarray(y_true)
    probs_array = np.asarray(probs)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_ids = np.digitize(probs_array, bin_edges[1:-1], right=True)

    bin_accuracies = np.full(n_bins, np.nan, dtype=float)
    bin_confidences = np.full(n_bins, np.nan, dtype=float)
    bin_sizes = np.zeros(n_bins, dtype=float)

    for bin_index in range(n_bins):
        mask = bin_ids == bin_index
        bin_sizes[bin_index] = mask.sum()
        if not np.any(mask):
            continue

        bin_accuracies[bin_index] = y_true_array[mask].mean()
        bin_confidences[bin_index] = probs_array[mask].mean()

    return bin_centers, bin_accuracies, bin_confidences, bin_sizes


def validation_auc(
    model: lgb.LGBMClassifier, X_val: pd.DataFrame, y_val: pd.Series
) -> float:
    """Compute validation ROC AUC for a fitted model."""

    return float(roc_auc_score(y_val, get_probs(model, X_val)))
