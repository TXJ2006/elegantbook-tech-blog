"""Reproducible experiments for
Change of Measure Arguments in Bandit Lower Bounds.

The script creates seven figures and one CSV file in the same directory.
All random experiments use a fixed NumPy seed.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binom, norm

OUTPUT_DIR = Path(__file__).resolve().parent
SEED = 20260620


def bernoulli_kl(p: np.ndarray | float, q: np.ndarray | float) -> np.ndarray | float:
    """KL(Bernoulli(p) || Bernoulli(q)), with stable clipping."""
    eps = np.finfo(float).eps
    p_arr = np.clip(np.asarray(p, dtype=float), eps, 1.0 - eps)
    q_arr = np.clip(np.asarray(q, dtype=float), eps, 1.0 - eps)
    value = p_arr * np.log(p_arr / q_arr) + (1.0 - p_arr) * np.log(
        (1.0 - p_arr) / (1.0 - q_arr)
    )
    if np.ndim(value) == 0:
        return float(value)
    return value


def bernoulli_llr(reward: np.ndarray, p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """log p(X)/q(X) for Bernoulli rewards and possibly arm-dependent means."""
    return reward * np.log(p / q) + (1.0 - reward) * np.log((1.0 - p) / (1.0 - q))


def save_figure(fig: plt.Figure, stem: str) -> None:
    fig.savefig(OUTPUT_DIR / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / f"{stem}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def experiment_evidence_paths(rng: np.random.Generator, rows: list[dict[str, object]]) -> None:
    p, q = 0.55, 0.45
    horizon = 400
    paths = 8
    rewards = rng.binomial(1, p, size=(paths, horizon)).astype(float)
    increments = bernoulli_llr(rewards, np.full_like(rewards, p), np.full_like(rewards, q))
    cumulative = np.cumsum(increments, axis=1)
    expected = np.arange(1, horizon + 1) * bernoulli_kl(p, q)

    fig = plt.figure(figsize=(7.2, 4.5))
    ax = fig.add_subplot(111)
    t = np.arange(1, horizon + 1)
    for path in cumulative:
        ax.plot(t, path, linewidth=1.0, alpha=0.72)
    ax.plot(t, expected, linewidth=2.4, linestyle="--", label=r"$t\,\mathrm{kl}(p,q)$")
    ax.axhline(0.0, linewidth=0.8)
    ax.set_xlabel("number of observations")
    ax.set_ylabel("cumulative log-likelihood ratio")
    ax.set_title("Evidence is noisy, but its mean drift is KL")
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    save_figure(fig, "evidence_paths")

    rows.append(
        {
            "experiment": "evidence_paths",
            "x": horizon,
            "setting": f"Bernoulli({p}) vs Bernoulli({q})",
            "metric": "mean_final_llr",
            "value": float(cumulative[:, -1].mean()),
        }
    )
    rows.append(
        {
            "experiment": "evidence_paths",
            "x": horizon,
            "setting": f"Bernoulli({p}) vs Bernoulli({q})",
            "metric": "theoretical_mean_final_llr",
            "value": float(expected[-1]),
        }
    )


def exact_lrt_error_sum(n: int, p: float, q: float) -> float:
    """Sum of the two errors for the equal-prior likelihood-ratio test."""
    successes = np.arange(n + 1)
    llr = successes * math.log(p / q) + (n - successes) * math.log((1.0 - p) / (1.0 - q))
    choose_p = llr >= 0.0
    error_under_p = binom.pmf(successes, n, p)[~choose_p].sum()
    error_under_q = binom.pmf(successes, n, q)[choose_p].sum()
    return float(error_under_p + error_under_q)


def experiment_testing_bound(rows: list[dict[str, object]]) -> None:
    p, q = 0.40, 0.60
    sample_sizes = np.array([5, 10, 20, 40, 80, 160, 240])
    exact_errors = np.array([exact_lrt_error_sum(int(n), p, q) for n in sample_sizes])
    bh_bounds = 0.5 * np.exp(-sample_sizes * bernoulli_kl(p, q))

    fig = plt.figure(figsize=(7.2, 4.5))
    ax = fig.add_subplot(111)
    ax.semilogy(sample_sizes, exact_errors, marker="o", label="optimal likelihood-ratio test")
    ax.semilogy(sample_sizes, bh_bounds, marker="s", linestyle="--", label="Bretagnolle-Huber lower bound")
    ax.set_xlabel("sample size n")
    ax.set_ylabel("sum of the two error probabilities")
    ax.set_title("Nearby models cannot be separated faster than their information allows")
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    save_figure(fig, "testing_error_bound")

    for n, exact, bound in zip(sample_sizes, exact_errors, bh_bounds):
        rows.extend(
            [
                {
                    "experiment": "testing_bound",
                    "x": int(n),
                    "setting": f"Bernoulli({p}) vs Bernoulli({q})",
                    "metric": "exact_lrt_error_sum",
                    "value": float(exact),
                },
                {
                    "experiment": "testing_bound",
                    "x": int(n),
                    "setting": f"Bernoulli({p}) vs Bernoulli({q})",
                    "metric": "bretagnolle_huber_bound",
                    "value": float(bound),
                },
            ]
        )


def simulate_ucb_information(
    rng: np.random.Generator,
    horizon: int,
    replications: int,
    true_means: np.ndarray,
    alternative_means: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Run UCB under the true environment and return counts and total LLR."""
    arms = len(true_means)
    counts = np.zeros((replications, arms), dtype=np.int64)
    sums = np.zeros((replications, arms), dtype=float)
    total_llr = np.zeros(replications, dtype=float)
    indices = np.arange(replications)

    # Pull each arm once.
    for arm in range(arms):
        rewards = rng.binomial(1, true_means[arm], size=replications).astype(float)
        counts[:, arm] += 1
        sums[:, arm] += rewards
        total_llr += bernoulli_llr(
            rewards,
            np.full(replications, true_means[arm]),
            np.full(replications, alternative_means[arm]),
        )

    for t in range(arms, horizon):
        means = sums / counts
        bonus = np.sqrt(2.0 * np.log(t + 1.0) / counts)
        actions = np.argmax(means + bonus, axis=1)
        arm_means = true_means[actions]
        alt_means = alternative_means[actions]
        rewards = rng.binomial(1, arm_means).astype(float)
        counts[indices, actions] += 1
        sums[indices, actions] += rewards
        total_llr += bernoulli_llr(rewards, arm_means, alt_means)

    return counts, total_llr


def experiment_adaptive_information(rng: np.random.Generator, rows: list[dict[str, object]]) -> None:
    true_means = np.array([0.62, 0.50])
    alternative_means = np.array([0.62, 0.68])
    horizons = np.array([50, 100, 200, 400, 800, 1200])
    replications = 12000
    empirical_llr: list[float] = []
    accounted_information: list[float] = []
    avg_arm2_counts: list[float] = []
    arm_kls = np.asarray(bernoulli_kl(true_means, alternative_means))

    for horizon in horizons:
        counts, total_llr = simulate_ucb_information(
            rng,
            int(horizon),
            replications,
            true_means,
            alternative_means,
        )
        average_counts = counts.mean(axis=0)
        lhs = float(total_llr.mean())
        rhs = float(np.dot(average_counts, arm_kls))
        empirical_llr.append(lhs)
        accounted_information.append(rhs)
        avg_arm2_counts.append(float(average_counts[1]))
        rows.extend(
            [
                {
                    "experiment": "adaptive_information",
                    "x": int(horizon),
                    "setting": "UCB: true=(0.62,0.50), alternative=(0.62,0.68)",
                    "metric": "empirical_mean_log_likelihood_ratio",
                    "value": lhs,
                },
                {
                    "experiment": "adaptive_information",
                    "x": int(horizon),
                    "setting": "UCB: true=(0.62,0.50), alternative=(0.62,0.68)",
                    "metric": "sum_expected_pulls_times_arm_kl",
                    "value": rhs,
                },
                {
                    "experiment": "adaptive_information",
                    "x": int(horizon),
                    "setting": "UCB: true=(0.62,0.50), alternative=(0.62,0.68)",
                    "metric": "average_pulls_arm_2",
                    "value": float(average_counts[1]),
                },
            ]
        )

    fig = plt.figure(figsize=(7.2, 4.5))
    ax = fig.add_subplot(111)
    ax.plot(horizons, empirical_llr, marker="o", label="empirical mean log-likelihood ratio")
    ax.plot(horizons, accounted_information, marker="s", linestyle="--", label=r"$\sum_a \mathbb{E}[N_a]D(\nu_a,\nu'_a)$")
    ax.set_xlabel("horizon")
    ax.set_ylabel("information")
    ax.set_title("Adaptive sampling changes the counts, not the information ledger")
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    save_figure(fig, "adaptive_information_identity")

    fig = plt.figure(figsize=(7.2, 4.5))
    ax = fig.add_subplot(111)
    ax.plot(horizons, avg_arm2_counts, marker="o")
    ax.set_xlabel("horizon")
    ax.set_ylabel("average pulls of the distinguishing arm")
    ax.set_title("UCB gathers evidence by revisiting the arm changed in the alternative")
    ax.grid(alpha=0.2)
    save_figure(fig, "distinguishing_arm_counts")


def experiment_gaussian_allocation(rng: np.random.Generator, rows: list[dict[str, object]]) -> None:
    mu1, mu2, sigma = 0.20, 0.00, 1.0
    total_budget = 400
    replications = 150000
    weights = np.arange(0.05, 1.00, 0.05)
    empirical_errors: list[float] = []
    exact_errors: list[float] = []
    information_rates: list[float] = []

    for weight in weights:
        n1 = max(1, int(round(total_budget * weight)))
        n2 = total_budget - n1
        mean1 = mu1 + sigma / math.sqrt(n1) * rng.standard_normal(replications)
        mean2 = mu2 + sigma / math.sqrt(n2) * rng.standard_normal(replications)
        empirical = float(np.mean(mean1 <= mean2))
        standard_error = sigma * math.sqrt(1.0 / n1 + 1.0 / n2)
        exact = float(norm.cdf(-(mu1 - mu2) / standard_error))
        effective_weight = n1 / total_budget
        information_rate = effective_weight * (1.0 - effective_weight) * (mu1 - mu2) ** 2 / (
            2.0 * sigma**2
        )
        empirical_errors.append(empirical)
        exact_errors.append(exact)
        information_rates.append(information_rate)
        rows.extend(
            [
                {
                    "experiment": "gaussian_allocation",
                    "x": effective_weight,
                    "setting": f"mu=({mu1},{mu2}), sigma={sigma}, budget={total_budget}",
                    "metric": "empirical_error_probability",
                    "value": empirical,
                },
                {
                    "experiment": "gaussian_allocation",
                    "x": effective_weight,
                    "setting": f"mu=({mu1},{mu2}), sigma={sigma}, budget={total_budget}",
                    "metric": "exact_error_probability",
                    "value": exact,
                },
                {
                    "experiment": "gaussian_allocation",
                    "x": effective_weight,
                    "setting": f"mu=({mu1},{mu2}), sigma={sigma}, budget={total_budget}",
                    "metric": "hardest_alternative_information_rate",
                    "value": information_rate,
                },
            ]
        )

    fig = plt.figure(figsize=(7.2, 4.5))
    ax = fig.add_subplot(111)
    ax.plot(weights, exact_errors, marker="o", label="exact Gaussian error")
    ax.scatter(weights, empirical_errors, s=28, label="Monte Carlo")
    ax.axvline(0.5, linestyle="--", linewidth=1.0, label="equal allocation")
    ax.set_xlabel("fraction of samples assigned to arm 1")
    ax.set_ylabel("probability of recommending the wrong arm")
    ax.set_title("A poor allocation leaves one side of the comparison too noisy")
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    save_figure(fig, "gaussian_allocation_error")

    fig = plt.figure(figsize=(7.2, 4.5))
    ax = fig.add_subplot(111)
    ax.plot(weights, information_rates, marker="o")
    ax.axvline(0.5, linestyle="--", linewidth=1.0)
    ax.set_xlabel("fraction of samples assigned to arm 1")
    ax.set_ylabel("information rate against the hardest alternative")
    ax.set_title("The lower-bound game selects equal allocation in the symmetric Gaussian case")
    ax.grid(alpha=0.2)
    save_figure(fig, "gaussian_information_rate")


def experiment_alternative_geometry(rows: list[dict[str, object]]) -> None:
    mu1, mu2, sigma = 0.20, 0.00, 1.0
    means = np.linspace(-0.08, 0.28, 361)
    weights: Iterable[float] = (0.20, 0.50, 0.80)

    fig = plt.figure(figsize=(7.2, 4.5))
    ax = fig.add_subplot(111)
    for weight in weights:
        objective = (
            weight * (mu1 - means) ** 2 + (1.0 - weight) * (mu2 - means) ** 2
        ) / (2.0 * sigma**2)
        minimizer = weight * mu1 + (1.0 - weight) * mu2
        minimum = weight * (1.0 - weight) * (mu1 - mu2) ** 2 / (2.0 * sigma**2)
        ax.plot(means, objective, label=f"allocation w={weight:.1f}")
        ax.scatter([minimizer], [minimum], s=35)
        rows.extend(
            [
                {
                    "experiment": "alternative_geometry",
                    "x": weight,
                    "setting": f"Gaussian mu=({mu1},{mu2}), sigma={sigma}",
                    "metric": "hardest_common_mean",
                    "value": minimizer,
                },
                {
                    "experiment": "alternative_geometry",
                    "x": weight,
                    "setting": f"Gaussian mu=({mu1},{mu2}), sigma={sigma}",
                    "metric": "minimum_information_rate",
                    "value": minimum,
                },
            ]
        )
    ax.set_xlabel("common mean m in the boundary alternative")
    ax.set_ylabel(r"$wD(\mu_1,m)+(1-w)D(\mu_2,m)$")
    ax.set_title("The hardest alternative moves toward the least-observed arm")
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    save_figure(fig, "gaussian_alternative_geometry")


def write_results(rows: list[dict[str, object]]) -> None:
    output = OUTPUT_DIR / "change_of_measure_results.csv"
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["experiment", "x", "setting", "metric", "value"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    rng = np.random.default_rng(SEED)
    rows: list[dict[str, object]] = []
    experiment_evidence_paths(rng, rows)
    experiment_testing_bound(rows)
    experiment_adaptive_information(rng, rows)
    experiment_gaussian_allocation(rng, rows)
    experiment_alternative_geometry(rows)
    write_results(rows)
    print(f"Wrote figures and results to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
