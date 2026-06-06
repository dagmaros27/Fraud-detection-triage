"""Routing strategies and system-level evaluation for fraud triage."""

from __future__ import annotations

from typing import Any

import numpy as np


def strategy_model_alone(probs, threshold: float = 0.5) -> dict[str, Any]:
    """Route all transactions automatically using a fixed model threshold.

    Parameters
    ----------
    probs:
        Fraud probabilities.
    threshold:
        Decision threshold for blocking a transaction.

    Returns
    -------
    dict[str, Any]
        Strategy outputs including final model decisions and routing labels.
    """

    probs_array = np.asarray(probs)
    decisions = np.where(probs_array >= threshold, "block", "approve")
    routing = np.where(probs_array >= threshold, "auto_block", "auto_approve")
    return {
        "decisions": decisions,
        "routing": routing,
        "escalation_rate": 0.0,
        "auto_rate": 1.0,
    }


def strategy_confidence_threshold(
    probs,
    low_thresh: float = 0.05,
    high_thresh: float = 0.80,
) -> dict[str, Any]:
    """Route transactions using fixed confidence bands.

    Parameters
    ----------
    probs:
        Fraud probabilities.
    low_thresh:
        Lower threshold for automatic approval.
    high_thresh:
        Upper threshold for automatic blocking.

    Returns
    -------
    dict[str, Any]
        Decisions and summary rates for the confidence-threshold strategy.
    """

    probs_array = np.asarray(probs)
    decisions = np.full(len(probs_array), "escalate", dtype=object)
    decisions[probs_array < low_thresh] = "auto_approve"
    decisions[probs_array > high_thresh] = "auto_block"

    escalation_mask = decisions == "escalate"
    return {
        "decisions": decisions,
        "escalation_rate": float(escalation_mask.mean()),
        "escalation_indices": np.flatnonzero(escalation_mask),
        "auto_approve_rate": float((decisions == "auto_approve").mean()),
        "auto_block_rate": float((decisions == "auto_block").mean()),
    }


def strategy_capacity_aware(
    probs,
    y_true,
    routing_conformal,
    daily_budget: int,
    arrival_order=None,
    c_escalate: float = 10.0,
    c_verify: float = 2.0,
    c_auto: float = 0.1,
) -> dict[str, Any]:
    """Implement capacity-aware routing with a simple shadow-pricing policy.

    Parameters
    ----------
    probs:
        Calibrated fraud probabilities.
    y_true:
        Ground-truth labels. Used only for alignment and analysis.
    routing_conformal:
        Three-zone routing labels from the conformal stage.
    daily_budget:
        Maximum number of analyst escalations allowed.
    arrival_order:
        Optional permutation defining transaction arrival order.
    c_escalate:
        Cost of an analyst escalation.
    c_verify:
        Cost of a verification step.
    c_auto:
        Cost of an automatic decision.

    Returns
    -------
    dict[str, Any]
        Routing decisions, rates, shadow prices, and budget usage.
    """

    probs_array = np.asarray(probs)
    _ = np.asarray(y_true)
    routing_array = np.asarray(routing_conformal, dtype=object)
    n = len(probs_array)

    if arrival_order is None:
        arrival_order = np.arange(n)
    else:
        arrival_order = np.asarray(arrival_order)

    # Complementarity gap is highest when the model is most uncertain.
    model_uncertainty = 1.0 - np.abs(2.0 * probs_array - 1.0)
    is_uncertain = (routing_array == "escalate").astype(float)
    delta_comp = model_uncertainty * is_uncertain
    if delta_comp.max() > 0:
        delta_comp = delta_comp / delta_comp.max()

    lambda_shadow = 0.0
    lambda_step = 1.0 / max(daily_budget, 1)

    budget_remaining = int(daily_budget)
    decisions = np.full(n, "auto_approve", dtype=object)
    shadow_prices = np.zeros(n, dtype=float)
    budget_trajectory = np.zeros(n, dtype=float)

    for idx in arrival_order:
        zone = routing_array[idx]
        shadow_prices[idx] = lambda_shadow

        if zone == "auto_approve":
            decisions[idx] = "auto_approve"
        elif zone == "auto_block":
            decisions[idx] = "auto_block"
        else:
            if budget_remaining > 0 and delta_comp[idx] > lambda_shadow:
                decisions[idx] = "escalate"
                budget_remaining -= 1
                lambda_shadow += lambda_step
            else:
                decisions[idx] = "verify"

        budget_trajectory[idx] = budget_remaining

    return {
        "decisions": decisions,
        "escalation_rate": float((decisions == "escalate").mean()),
        "verify_rate": float((decisions == "verify").mean()),
        "auto_approve_rate": float((decisions == "auto_approve").mean()),
        "auto_block_rate": float((decisions == "auto_block").mean()),
        "shadow_prices": shadow_prices,
        "budget_trajectory": budget_trajectory,
        "delta_comp": delta_comp,
        "budget_used": int(daily_budget - budget_remaining),
        "budget_total": int(daily_budget),
        "budget_utilization": float((daily_budget - budget_remaining) / max(daily_budget, 1)),
        "cost_params": {"auto": c_auto, "verify": c_verify, "escalate": c_escalate},
    }


def simulate_analyst(
    decisions,
    probs,
    y_true,
    analyst_error_rate: float = 0.15,
    verify_catch_rate: float = 0.60,
    random_state: int = 42,
) -> dict[str, np.ndarray]:
    """Simulate downstream outcomes for routing decisions.

    Parameters
    ----------
    decisions:
        Array of routing or final-decision labels.
    probs:
        Fraud probabilities.
    y_true:
        Ground-truth labels.
    analyst_error_rate:
        Error rate for analyst escalations.
    verify_catch_rate:
        Fraud catch rate for verification checks.
    random_state:
        Random seed for simulation reproducibility.

    Returns
    -------
    dict[str, np.ndarray]
        Final decisions plus error and confusion-indicator arrays.
    """

    decisions_array = np.asarray(decisions, dtype=object)
    probs_array = np.asarray(probs)
    y_array = np.asarray(y_true, dtype=int)
    rng = np.random.default_rng(random_state)

    final_decisions = np.empty(len(decisions_array), dtype=object)

    for index, decision in enumerate(decisions_array):
        if decision in {"block", "approve"}:
            final_decisions[index] = decision
        elif decision == "auto_approve":
            final_decisions[index] = "approve"
        elif decision == "auto_block":
            final_decisions[index] = "block"
        elif decision == "auto_decide":
            final_decisions[index] = "block" if probs_array[index] >= 0.5 else "approve"
        elif decision == "verify":
            if y_array[index] == 1:
                caught = rng.random() < verify_catch_rate
                final_decisions[index] = "block" if caught else "approve"
            else:
                final_decisions[index] = "approve"
        elif decision == "escalate":
            analyst_correct = rng.random() < (1.0 - analyst_error_rate)
            if analyst_correct:
                final_decisions[index] = "block" if y_array[index] == 1 else "approve"
            else:
                final_decisions[index] = "approve" if y_array[index] == 1 else "block"
        else:
            raise ValueError(f"Unknown decision label: {decision}")

    should_block = y_array == 1
    predicted_block = final_decisions == "block"

    errors = predicted_block != should_block
    false_positives = (y_array == 0) & predicted_block
    false_negatives = (y_array == 1) & (~predicted_block)

    return {
        "final_decisions": final_decisions,
        "errors": errors,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
    }


def compute_system_metrics(
    decisions_dict,
    simulation_result,
    y_true,
    probs,
    costs,
) -> dict[str, float]:
    """Compute risk and cost metrics for a routing strategy.

    Parameters
    ----------
    decisions_dict:
        Strategy output dictionary containing a ``decisions`` array.
    simulation_result:
        Output from :func:`simulate_analyst`.
    y_true:
        Ground-truth labels.
    probs:
        Fraud probabilities.
    costs:
        Cost dictionary with keys ``auto``, ``verify``, and ``escalate``.

    Returns
    -------
    dict[str, float]
        System-level performance metrics for the strategy.
    """

    decisions = np.asarray(decisions_dict["decisions"], dtype=object)
    y_array = np.asarray(y_true, dtype=int)
    _ = np.asarray(probs)

    auto_mask = np.isin(
        decisions,
        ["auto_approve", "auto_block", "auto_decide", "block", "approve"],
    )
    verify_mask = decisions == "verify"
    escalate_mask = decisions == "escalate"

    total_cost = (
        auto_mask.sum() * costs["auto"]
        + verify_mask.sum() * costs["verify"]
        + escalate_mask.sum() * costs["escalate"]
    )

    errors = np.asarray(simulation_result["errors"], dtype=bool)
    false_positives = np.asarray(simulation_result["false_positives"], dtype=bool)
    false_negatives = np.asarray(simulation_result["false_negatives"], dtype=bool)

    total_legitimate = max(int((y_array == 0).sum()), 1)
    total_fraud = max(int((y_array == 1).sum()), 1)
    total_correct = max(int((~errors).sum()), 1)

    escalation_rate = float(escalate_mask.mean())
    coverage = 1.0 - escalation_rate

    return {
        "total_cost": float(total_cost),
        "risk": float(errors.mean()),
        "false_positive_rate": float(false_positives.sum() / total_legitimate),
        "false_negative_rate": float(false_negatives.sum() / total_fraud),
        "escalation_rate": escalation_rate,
        "coverage": coverage,
        "cost_per_correct": float(total_cost / total_correct),
    }
