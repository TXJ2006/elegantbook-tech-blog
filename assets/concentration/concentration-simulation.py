"""Experiments for the lecture note
Concentration Bounds in Bandit Analysis.

The script generates three figures and one CSV file:
  1. fixed_time_hoeffding.pdf/png
  2. confidence_radius_widths.pdf/png
  3. ucb_regret_curves.pdf/png
  4. concentration_results.csv

Only numpy, pandas, and matplotlib are used.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent


def hoeffding_radius(n: np.ndarray | float, delta: float) -> np.ndarray | float:
    """Two-sided Hoeffding confidence radius for variables in [0,1]."""
    return np.sqrt(np.log(2.0 / delta) / (2.0 * np.asarray(n)))


def bernstein_radius_oracle(n: np.ndarray | float, delta: float, var: float) -> np.ndarray | float:
    """A simple variance-aware Bernstein-style radius."""
    x = np.log(2.0 / delta)
    n = np.asarray(n)
    return np.sqrt(2.0 * var * x / n) + x / (3.0 * n)


def fixed_time_coverage(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    p = 0.30
    delta = 0.05
    trials = 50_000
    ns = np.array([10, 20, 50, 100, 200, 500, 1000])
    rows = []
    empirical = []
    hoeffding_bounds = []
    for n in ns:
        samples = rng.binomial(1, p, size=(trials, n))
        means = samples.mean(axis=1)
        r = hoeffding_radius(n, delta)
        miss = np.mean(np.abs(means - p) > r)
        rows.append({
            "experiment": "fixed_time_coverage",
            "n": int(n),
            "delta": delta,
            "p": p,
            "hoeffding_radius": float(r),
            "empirical_miss_probability": float(miss),
        })
        empirical.append(miss)
        hoeffding_bounds.append(delta)

    fig, ax = plt.subplots(figsize=(6.5, 4.1))
    ax.plot(ns, empirical, marker="o", label="Empirical miss probability")
    ax.plot(ns, hoeffding_bounds, linestyle="--", label="Nominal delta = 0.05")
    ax.set_xscale("log")
    ax.set_xlabel("sample size n")
    ax.set_ylabel("probability")
    ax.set_title("Fixed-time Hoeffding intervals are conservative")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "fixed_time_hoeffding.pdf")
    fig.savefig(OUT / "fixed_time_hoeffding.png", dpi=220)
    plt.close(fig)
    return pd.DataFrame(rows)


def radius_width_plot() -> pd.DataFrame:
    delta = 0.05
    p = 0.05
    var = p * (1.0 - p)
    ns = np.arange(5, 1001)
    h = hoeffding_radius(ns, delta)
    b = bernstein_radius_oracle(ns, delta, var)

    fig, ax = plt.subplots(figsize=(6.5, 4.1))
    ax.plot(ns, h, label="Hoeffding radius")
    ax.plot(ns, b, label="Bernstein radius using variance")
    ax.set_xlabel("sample size n")
    ax.set_ylabel("radius")
    ax.set_title("Variance-aware error bars can be much shorter")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "confidence_radius_widths.pdf")
    fig.savefig(OUT / "confidence_radius_widths.png", dpi=220)
    plt.close(fig)

    return pd.DataFrame({
        "experiment": "radius_widths",
        "n": ns,
        "delta": delta,
        "p": p,
        "variance": var,
        "hoeffding_radius": h,
        "bernstein_radius": b,
    })


def run_bandit(policy: str, p: np.ndarray, T: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    K = len(p)
    counts = np.zeros(K, dtype=int)
    sums = np.zeros(K, dtype=float)
    sq_sums = np.zeros(K, dtype=float)
    regret = np.zeros(T, dtype=float)
    p_star = float(np.max(p))
    cumulative_regret = 0.0

    for t in range(min(K, T)):
        a = t
        r = rng.binomial(1, p[a])
        counts[a] += 1
        sums[a] += r
        sq_sums[a] += r * r
        cumulative_regret += p_star - p[a]
        regret[t] = cumulative_regret

    for t in range(K, T):
        means = sums / np.maximum(counts, 1)
        if policy == "Greedy":
            a = int(np.argmax(means))
        elif policy == "UCB-Hoeffding":
            bonus = np.sqrt(2.0 * np.log(t + 1.0) / counts)
            a = int(np.argmax(means + bonus))
        elif policy == "UCB-Bernstein":
            variances = np.zeros(K)
            for i in range(K):
                n = counts[i]
                if n <= 1:
                    variances[i] = 0.25
                else:
                    m = means[i]
                    variances[i] = max(0.0, sq_sums[i] / n - m * m)
            log_term = np.log(t + 1.0)
            bonus = np.sqrt(2.0 * variances * log_term / counts) + 3.0 * log_term / counts
            a = int(np.argmax(means + bonus))
        elif policy == "Thompson":
            alpha = sums + 1.0
            beta = counts - sums + 1.0
            a = int(np.argmax(rng.beta(alpha, beta)))
        else:
            raise ValueError(f"Unknown policy: {policy}")

        r = rng.binomial(1, p[a])
        counts[a] += 1
        sums[a] += r
        sq_sums[a] += r * r
        cumulative_regret += p_star - p[a]
        regret[t] = cumulative_regret

    return regret, counts


def bandit_experiment(seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    p = np.array([0.01, 0.03, 0.05, 0.07])
    T = 3000
    trials = 150
    policies = ["Greedy", "UCB-Hoeffding", "UCB-Bernstein", "Thompson"]
    mean_regrets = {}
    rows = []

    for policy in policies:
        regrets = []
        counts_all = []
        for _ in range(trials):
            regret, counts = run_bandit(policy, p, T, rng)
            regrets.append(regret)
            counts_all.append(counts)
        regrets = np.asarray(regrets)
        counts_all = np.asarray(counts_all)
        mean_regrets[policy] = regrets.mean(axis=0)
        row = {"experiment": "bandit_regret", "policy": policy, "T": T, "trials": trials,
               "final_regret": float(mean_regrets[policy][-1])}
        for i, val in enumerate(counts_all.mean(axis=0)):
            row[f"mean_pulls_arm_{i}"] = float(val)
        rows.append(row)

    fig, ax = plt.subplots(figsize=(6.7, 4.3))
    x = np.arange(1, T + 1)
    for policy in policies:
        ax.plot(x, mean_regrets[policy], label=policy)
    ax.set_xlabel("round t")
    ax.set_ylabel("average cumulative regret")
    ax.set_title("Concentration turns uncertainty into action")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "ucb_regret_curves.pdf")
    fig.savefig(OUT / "ucb_regret_curves.png", dpi=220)
    plt.close(fig)

    return pd.DataFrame(rows)


def main() -> None:
    tables = [fixed_time_coverage(), radius_width_plot(), bandit_experiment()]
    out = pd.concat(tables, ignore_index=True, sort=False)
    out.to_csv(OUT / "concentration_results.csv", index=False)
    print(out.tail(12).to_string(index=False))


if __name__ == "__main__":
    main()
