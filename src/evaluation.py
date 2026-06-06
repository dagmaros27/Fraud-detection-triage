"""Evaluation helpers for routing performance and fairness analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_risk_coverage_curve(probs, y_true, thresholds=None):
    """Compute a selective risk-coverage curve over symmetric confidence bands.

    Parameters
    ----------
    probs:
        Fraud probabilities.
    y_true:
        Ground-truth binary labels.
    thresholds:
        Optional array of band-width thresholds. When omitted, uses
        ``np.linspace(0.01, 0.99, 100)``.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        Arrays of coverage values, risk values, and thresholds.
    """

    probs_array = np.asarray(probs, dtype=float)
    y_array = np.asarray(y_true, dtype=int)
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 100)
    thresholds = np.asarray(thresholds, dtype=float)

    coverages = np.zeros(len(thresholds), dtype=float)
    risks = np.full(len(thresholds), np.nan, dtype=float)

    for idx, threshold in enumerate(thresholds):
        low = threshold / 2.0
        high = 1.0 - threshold / 2.0
        auto_mask = (probs_array < low) | (probs_array > high)
        coverages[idx] = auto_mask.mean()
        if not np.any(auto_mask):
            continue
        preds = (probs_array[auto_mask] >= 0.5).astype(int)
        risks[idx] = np.mean(preds != y_array[auto_mask])

    return coverages, risks, thresholds


def compute_capacity_trajectory(results_s3):
    """Extract intraday shadow-price and capacity dynamics from strategy 3.

    Parameters
    ----------
    results_s3:
        Result dictionary returned by ``strategy_capacity_aware``.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        Normalized time steps, shadow prices, remaining budget, cumulative
        escalations, and cumulative verifications.
    """

    shadow_prices = np.asarray(results_s3["shadow_prices"], dtype=float)
    budget_remaining = np.asarray(results_s3["budget_trajectory"], dtype=float)
    decisions = np.asarray(results_s3["decisions"], dtype=object)
    n = len(shadow_prices)
    if n == 0:
        return (
            np.array([]),
            np.array([]),
            np.array([]),
            np.array([]),
            np.array([]),
        )

    time_steps = np.linspace(0.0, 1.0, n)
    cumulative_escalations = np.cumsum(decisions == "escalate")
    cumulative_verifications = np.cumsum(decisions == "verify")
    return (
        time_steps,
        shadow_prices,
        budget_remaining,
        cumulative_escalations,
        cumulative_verifications,
    )


def compute_fairness_metrics(decisions, y_true, proxy_groups, group_col):
    """Compute routing and error metrics for each proxy group.

    Parameters
    ----------
    decisions:
        A dataframe containing at least ``routing_decision`` and
        ``final_decision`` columns, or an array-like of final decisions.
    y_true:
        Ground-truth labels.
    proxy_groups:
        Dataframe of proxy group assignments.
    group_col:
        Group column to analyze.

    Returns
    -------
    pd.DataFrame
        One row per group value with routing and error metrics.
    """

    y_array = np.asarray(y_true, dtype=int)
    groups = pd.Series(proxy_groups[group_col]).reset_index(drop=True)

    if isinstance(decisions, pd.DataFrame):
        routing_decision = decisions["routing_decision"].reset_index(drop=True)
        final_decision = decisions["final_decision"].reset_index(drop=True)
    else:
        routing_decision = pd.Series(decisions).reset_index(drop=True)
        final_decision = pd.Series(decisions).reset_index(drop=True)

    data = pd.DataFrame(
        {
            "group": groups,
            "routing_decision": routing_decision,
            "final_decision": final_decision,
            "y_true": y_array,
        }
    )

    rows = []
    for group_value, group_df in data.groupby("group"):
        legit = group_df["y_true"] == 0
        fraud = group_df["y_true"] == 1
        predicted_block = group_df["final_decision"] == "block"

        rows.append(
            {
                "group": group_value,
                "escalation_rate": (group_df["routing_decision"] == "escalate").mean(),
                "verify_rate": (group_df["routing_decision"] == "verify").mean(),
                "auto_approve_rate": (group_df["routing_decision"] == "auto_approve").mean(),
                "false_negative_rate": (
                    ((group_df["final_decision"] == "approve") & fraud).sum() / max(fraud.sum(), 1)
                ),
                "false_positive_rate": (
                    (predicted_block & legit).sum() / max(legit.sum(), 1)
                ),
            }
        )

    return pd.DataFrame(rows).sort_values("group").reset_index(drop=True)


def compute_complementarity_by_group(probs, y_true, proxy_groups, group_col):
    """Approximate complementarity gains by proxy group.

    Parameters
    ----------
    probs:
        Fraud probabilities.
    y_true:
        Ground-truth labels.
    proxy_groups:
        Dataframe of proxy group assignments.
    group_col:
        Group column to analyze.

    Returns
    -------
    pd.DataFrame
        Dataframe with model, human, and team risk estimates plus
        ``delta_comp`` for each group.
    """

    probs_array = np.asarray(probs, dtype=float)
    y_array = np.asarray(y_true, dtype=int)
    groups = pd.Series(proxy_groups[group_col]).reset_index(drop=True)

    model_pred = (probs_array >= 0.5).astype(int)
    model_error = (model_pred != y_array).astype(float)
    model_uncertainty = 1.0 - np.abs(2.0 * probs_array - 1.0)
    human_risk = np.full(len(probs_array), 0.15)
    team_risk = np.where(model_uncertainty > 0.4, 0.10, model_error)

    data = pd.DataFrame(
        {
            "group": groups,
            "model_risk": model_error,
            "human_risk": human_risk,
            "team_risk": team_risk,
        }
    )

    summary = (
        data.groupby("group", as_index=False)
        .agg(
            model_risk=("model_risk", "mean"),
            human_risk=("human_risk", "mean"),
            team_risk=("team_risk", "mean"),
        )
        .sort_values("group")
        .reset_index(drop=True)
    )
    summary["delta_comp"] = np.minimum(summary["model_risk"], summary["human_risk"]) - summary["team_risk"]
    return summary
