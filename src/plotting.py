"""Plotting helpers for fraud modeling and calibration analysis."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import gaussian_kde

from src.model import compute_ece, reliability_diagram_data


def plot_reliability_diagram(
    y_true_before,
    probs_before,
    y_true_after,
    probs_after,
    save_path,
    temperature: float | None = None,
) -> None:
    """Plot reliability diagrams before and after temperature scaling."""

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    before_data = reliability_diagram_data(y_true_before, probs_before)
    after_data = reliability_diagram_data(y_true_after, probs_after)
    ece_before = compute_ece(y_true_before, probs_before)
    ece_after = compute_ece(y_true_after, probs_after)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Model Calibration: Before and After Temperature Scaling", fontsize=12)

    panel_specs = [
        (axes[0], before_data, ece_before, "Before Calibration"),
        (axes[1], after_data, ece_after, "After Temperature Scaling"),
    ]

    for ax, (bin_centers, bin_accuracies, bin_confidences, bin_sizes), ece_value, panel_title in panel_specs:
        max_size = max(bin_sizes.max(), 1.0)
        if len(bin_centers) > 1:
            gaps = np.diff(np.sort(bin_centers))
            bar_width = max(min(float(np.min(gaps)) * 0.85, 0.08), 0.01)
        else:
            bar_width = 0.08

        if len(bin_centers) > 0:
            x_min = float(bin_centers.min())
            x_max = float(bin_centers.max())
            padding = max((x_max - x_min) * 0.15, 0.02)
            x_left = max(0.0, x_min - padding)
            x_right = min(1.0, x_max + padding)
        else:
            x_left, x_right = 0.0, 1.0

        for center, size in zip(bin_centers, bin_sizes):
            alpha = 0.08 + 0.22 * (size / max_size)
            ax.axvspan(
                max(x_left, center - bar_width / 2),
                min(x_right, center + bar_width / 2),
                color="lightgray",
                alpha=alpha,
                zorder=0,
            )

        accuracy_values = np.nan_to_num(bin_accuracies, nan=0.0)
        valid_confidences = ~np.isnan(bin_confidences)

        ax.bar(
            bin_centers,
            accuracy_values,
            width=bar_width,
            color="steelblue",
            alpha=0.7,
            label="Bin accuracy",
            zorder=2,
        )
        ax.plot(
            bin_centers[valid_confidences],
            bin_confidences[valid_confidences],
            linestyle="--",
            color="gray",
            linewidth=2,
            label="Mean confidence",
            zorder=3,
        )
        ax.plot(
            [0, 1],
            [0, 1],
            linestyle="--",
            color="red",
            linewidth=1.5,
            label="Perfect calibration",
            zorder=1,
        )
        ax.set_xlim(x_left, x_right)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Confidence")
        ax.set_ylabel("Accuracy")
        if panel_title == "After Temperature Scaling" and temperature is not None:
            ax.set_title(f"{panel_title} (ECE={ece_value:.3f}, T={temperature:.2f})", fontsize=12)
        else:
            ax.set_title(f"{panel_title} (ECE={ece_value:.3f})", fontsize=12)
        ax.legend(loc="lower right")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_score_distribution(y_true, probs_raw, probs_cal, save_path) -> None:
    """Plot raw and calibrated fraud score distributions by true class."""

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    y_true_array = np.asarray(y_true)
    probs_raw_array = np.asarray(probs_raw)
    probs_cal_array = np.asarray(probs_cal)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Fraud Score Distribution: Raw vs Calibrated", fontsize=12)

    panel_specs = [
        (axes[0], probs_raw_array, "Raw Fraud Probability"),
        (axes[1], probs_cal_array, "Calibrated Fraud Probability"),
    ]

    for ax, probabilities, xlabel in panel_specs:
        for label, color, alpha, name in [
            (0, "steelblue", 0.4, "Legitimate"),
            (1, "red", 0.6, "Fraud"),
        ]:
            class_probs = probabilities[y_true_array == label]
            if len(class_probs) == 0:
                continue

            if len(np.unique(class_probs)) > 1:
                x_grid = np.linspace(0, 1, 400)
                density = gaussian_kde(class_probs)(x_grid)
                ax.fill_between(x_grid, density, color=color, alpha=alpha, label=name)
                ax.plot(x_grid, density, color=color, linewidth=1.5)
            else:
                ax.axvline(float(class_probs[0]), color=color, alpha=max(alpha, 0.6), linewidth=2, label=name)

        ax.axvline(0.5, color="black", linestyle="--", linewidth=1.2, label="0.5 threshold")
        ax.set_xlim(0, 1)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Density")
        ax.legend()
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_conformal_analysis(
    nonconformity_scores,
    y_cal,
    probs_test,
    set_sizes_10,
    set_sizes_30,
    routing_30,
    tau_10,
    tau_30,
    save_path,
    low_thresh: float = 0.05,
    high_thresh: float = 0.80,
) -> None:
    """Plot conformal diagnostics and three-zone routing behavior."""

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    nonconformity_scores = np.asarray(nonconformity_scores)
    probs_test = np.asarray(probs_test)
    set_sizes_10 = np.asarray(set_sizes_10)
    set_sizes_30 = np.asarray(set_sizes_30)
    routing_30 = np.asarray(routing_30)
    y_cal = np.asarray(y_cal)

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    fig.suptitle("Conformal Routing Signal: Three-Zone Triage System", fontsize=11)

    legitimate_mask = y_cal == 0
    fraud_mask = y_cal == 1

    axes[0].hist(
        nonconformity_scores[legitimate_mask],
        bins=20,
        color="steelblue",
        alpha=0.5,
        label="Legitimate",
    )
    axes[0].hist(
        nonconformity_scores[fraud_mask],
        bins=20,
        color="red",
        alpha=0.6,
        label="Fraud",
    )
    axes[0].axvline(
        tau_10,
        color="black",
        linestyle="--",
        linewidth=1.5,
        label=fr"$\hat{{\tau}}_{{0.10}}$ = {tau_10:.3f}",
    )
    axes[0].axvline(
        tau_30,
        color="dimgray",
        linestyle=":",
        linewidth=1.5,
        label=fr"$\hat{{\tau}}_{{0.30}}$ = {tau_30:.3f}",
    )
    axes[0].set_xlabel("Nonconformity Score")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Nonconformity Score Distribution", fontsize=11)
    axes[0].legend()

    size_labels = np.array([1, 2])
    counts_10 = np.array([(set_sizes_10 == 1).sum(), (set_sizes_10 == 2).sum()])
    counts_30 = np.array([(set_sizes_30 == 1).sum(), (set_sizes_30 == 2).sum()])
    x = np.arange(len(size_labels))
    width = 0.35
    bars_10 = axes[1].bar(x - width / 2, counts_10, width=width, color="steelblue", label="α=0.10")
    bars_30 = axes[1].bar(x + width / 2, counts_30, width=width, color="coral", label="α=0.30")
    total_count = max(len(set_sizes_10), 1)
    for bars, counts in [(bars_10, counts_10), (bars_30, counts_30)]:
        for bar, count in zip(bars, counts):
            pct = 100.0 * count / total_count
            axes[1].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{pct:.1f}%",
                ha="center",
                va="bottom",
                fontsize=9,
            )
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(size_labels)
    axes[1].set_xlabel("Prediction Set Size")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Set Size Distribution", fontsize=11)
    axes[1].legend()

    zone_order = ["auto_approve", "auto_block", "auto_decide", "escalate"]
    zone_labels = ["Auto-approve", "Auto-block", "Auto-decide", "Escalate"]
    zone_colors = {
        "auto_approve": "steelblue",
        "auto_block": "red",
        "auto_decide": "green",
        "escalate": "coral",
    }
    zone_counts = np.array([(routing_30 == zone).sum() for zone in zone_order])
    y_positions = np.arange(len(zone_order))
    bars = axes[2].barh(
        y_positions,
        zone_counts,
        color=[zone_colors[zone] for zone in zone_order],
        alpha=0.85,
    )
    total_routed = max(len(routing_30), 1)
    for bar, count in zip(bars, zone_counts):
        pct = 100.0 * count / total_routed
        axes[2].text(
            bar.get_width(),
            bar.get_y() + bar.get_height() / 2,
            f" {count} ({pct:.1f}%)",
            va="center",
            fontsize=9,
        )
    axes[2].set_yticks(y_positions)
    axes[2].set_yticklabels(zone_labels)
    axes[2].set_xlabel("Count")
    axes[2].set_title("Three-Zone Routing (α=0.30)", fontsize=11)
    axes[2].invert_yaxis()

    bins = np.linspace(0, 1, 40)
    for zone in zone_order:
        zone_probs = probs_test[routing_30 == zone]
        if len(zone_probs) == 0:
            continue
        axes[3].hist(
            zone_probs,
            bins=bins,
            density=True,
            alpha=0.35,
            color=zone_colors[zone],
            label=zone.replace("_", " ").title(),
        )
    axes[3].axvspan(0, low_thresh, color="steelblue", alpha=0.08)
    axes[3].axvspan(high_thresh, 1, color="red", alpha=0.08)
    axes[3].axvline(low_thresh, color="steelblue", linestyle="--", linewidth=1.5)
    axes[3].axvline(high_thresh, color="red", linestyle="--", linewidth=1.5)
    axes[3].set_xlabel("Fraud Probability")
    axes[3].set_ylabel("Density")
    axes[3].set_title("Routing Signal by Score", fontsize=11)
    axes[3].legend(loc="upper center", fontsize=8)

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=10)
        ax.grid(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_risk_coverage_curve(coverages_list, risks_list, labels, save_path) -> None:
    """Plot risk-coverage tradeoff curves for multiple strategies."""

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    style_map = {
        "Model Alone": {"color": "gray", "linestyle": "--", "linewidth": 2},
        "Conf. Threshold": {"color": "coral", "linestyle": "-", "linewidth": 2},
        "Capacity-Aware": {"color": "steelblue", "linestyle": "-", "linewidth": 3},
    }

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.axvspan(0.85, 1.0, color="lightgray", alpha=0.2, zorder=0)
    ax.text(0.855, 0.98, "capacity constraint zone", transform=ax.transAxes, fontsize=9, va="top")

    for coverages, risks, label in zip(coverages_list, risks_list, labels):
        style = style_map.get(label, {"color": "black", "linestyle": "-", "linewidth": 2})
        coverages = np.asarray(coverages, dtype=float)
        risks = np.asarray(risks, dtype=float)
        valid = ~np.isnan(risks)
        ax.plot(coverages[valid], risks[valid], label=label, **style)
        if np.any(valid):
            ax.scatter(
                coverages[valid][-1],
                risks[valid][-1],
                color=style["color"],
                s=60,
                zorder=4,
            )
            ax.annotate(
                label,
                (coverages[valid][-1], risks[valid][-1]),
                xytext=(8, -8),
                textcoords="offset points",
                fontsize=9,
            )

    ax.set_xlabel("Coverage (fraction auto-decided)")
    ax.set_ylabel("Selective Risk (error rate)")
    ax.set_title("Risk-Coverage Tradeoff Across Routing Strategies")
    ax.legend(loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_capacity_trajectory(
    time_steps,
    shadow_prices,
    budget_remaining,
    cumulative_escalations,
    cumulative_verifications,
    daily_budget,
    save_path,
) -> None:
    """Plot intraday capacity-aware routing dynamics."""

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Capacity-Aware Routing: Intraday Dynamics", fontsize=12)

    axes[0].plot(time_steps, shadow_prices, color="steelblue", linewidth=2)
    axes[0].fill_between(time_steps, shadow_prices, color="steelblue", alpha=0.2)
    axes[0].axvline(0.5, color="black", linestyle="--", linewidth=1)
    axes[0].set_xlabel("Fraction of Day Elapsed")
    axes[0].set_ylabel("Shadow Price λ")
    axes[0].set_title("Shadow Price Trajectory")

    low_budget_mask = budget_remaining < 0.2 * daily_budget
    axes[1].plot(time_steps, budget_remaining, color="firebrick", linewidth=2)
    if np.any(low_budget_mask):
        axes[1].plot(
            np.asarray(time_steps)[low_budget_mask],
            np.asarray(budget_remaining)[low_budget_mask],
            color="red",
            linewidth=3,
        )
    axes[1].axhline(0, color="black", linestyle="--", linewidth=1)
    axes[1].set_xlabel("Fraction of Day Elapsed")
    axes[1].set_ylabel("Analyst Slots Remaining")
    axes[1].set_title("Analyst Capacity Over the Day")

    axes[2].plot(time_steps, cumulative_escalations, color="steelblue", linewidth=2, label="Escalations")
    axes[2].plot(time_steps, cumulative_verifications, color="coral", linewidth=2, label="Verifications")
    axes[2].set_xlabel("Fraction of Day Elapsed")
    axes[2].set_ylabel("Cumulative Count")
    axes[2].set_title("Escalation vs Verification Routing")
    axes[2].legend()

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_fairness_analysis(fairness_s2, fairness_s3, group_col, save_path) -> None:
    """Plot disparate deferral and error rates by proxy group."""

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fairness_s2 = fairness_s2.copy()
    fairness_s3 = fairness_s3.copy()
    groups = fairness_s2["group"].astype(str).tolist()
    x = np.arange(len(groups))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Fairness Analysis: {group_col}", fontsize=12)

    axes[0].bar(x - width / 2, fairness_s2["escalation_rate"], width=width, color="coral", label="Conf. Threshold")
    axes[0].bar(x + width / 2, fairness_s3["escalation_rate"], width=width, color="steelblue", label="Capacity-Aware")
    axes[0].axhline(fairness_s2["escalation_rate"].mean(), color="coral", linestyle="--", linewidth=1)
    axes[0].axhline(fairness_s3["escalation_rate"].mean(), color="steelblue", linestyle="--", linewidth=1)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(groups, rotation=20, ha="right")
    axes[0].set_ylabel("Escalation Rate")
    axes[0].set_title(f"Escalation Rate by {group_col} (Disparate Deferral Analysis)")
    axes[0].legend()

    axes[1].bar(x - width / 2, fairness_s2["false_negative_rate"], width=width, color="coral", label="Conf. Threshold")
    axes[1].bar(x + width / 2, fairness_s3["false_negative_rate"], width=width, color="steelblue", label="Capacity-Aware")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(groups, rotation=20, ha="right")
    axes[1].set_ylabel("False Negative Rate")
    axes[1].set_title(f"False Negative Rate by {group_col}")

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_summary_dashboard(results_s1, results_s2, results_s3, sim_s1, sim_s2, sim_s3, save_path) -> None:
    """Plot a multi-panel dashboard comparing the three routing strategies."""

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    strategy_labels = ["Model Alone", "Conf. Threshold", "Capacity-Aware"]
    colors = ["gray", "coral", "steelblue"]
    strategies = [results_s1, results_s2, results_s3]
    sims = [sim_s1, sim_s2, sim_s3]

    risks = [np.mean(sim["errors"]) for sim in sims]
    fns = [np.mean(sim["false_negatives"]) for sim in sims]

    def _cost(strategy):
        decisions = np.asarray(strategy["decisions"], dtype=object)
        return (
            np.isin(decisions, ["auto_approve", "auto_block", "auto_decide", "block", "approve"]).sum() * 0.1
            + (decisions == "verify").sum() * 2.0
            + (decisions == "escalate").sum() * 10.0
        )

    total_costs = [_cost(strategy) for strategy in strategies]
    correct_counts = [max(int((~np.asarray(sim["errors"])).sum()), 1) for sim in sims]
    cost_per_correct = [cost / correct for cost, correct in zip(total_costs, correct_counts)]

    routing_categories = ["auto_approve", "auto_block", "verify", "escalate"]
    routing_colors = {
        "auto_approve": "steelblue",
        "auto_block": "red",
        "verify": "green",
        "escalate": "coral",
    }

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("Strategy Comparison Dashboard", fontsize=12)

    axes[0, 0].bar(strategy_labels, risks, color=colors)
    axes[0, 0].set_title("Risk")

    axes[0, 1].bar(strategy_labels, total_costs, color=colors)
    axes[0, 1].set_title("Total Cost")

    axes[0, 2].bar(strategy_labels, fns, color=colors)
    axes[0, 2].set_title("False Negative Rate")

    bottom = np.zeros(len(strategy_labels))
    for category in routing_categories:
        values = [np.mean(np.asarray(strategy["decisions"], dtype=object) == category) for strategy in strategies]
        axes[1, 0].bar(strategy_labels, values, bottom=bottom, color=routing_colors[category], label=category)
        bottom += values
    axes[1, 0].set_title("Routing Decision Mix")
    axes[1, 0].legend()

    escalation_rates = [np.mean(np.asarray(strategy["decisions"], dtype=object) == "escalate") for strategy in strategies]
    axes[1, 1].scatter(escalation_rates, risks, s=np.array(total_costs) / 20.0, color=colors)
    for x, y, label in zip(escalation_rates, risks, strategy_labels):
        axes[1, 1].annotate(label, (x, y), xytext=(8, 8), textcoords="offset points")
    axes[1, 1].set_xlabel("Escalation Rate")
    axes[1, 1].set_ylabel("Risk")
    axes[1, 1].set_title("Escalation Rate vs Risk")

    axes[1, 2].bar(strategy_labels, cost_per_correct, color=colors)
    axes[1, 2].set_title("Cost per Correct")

    for ax in axes.ravel():
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
