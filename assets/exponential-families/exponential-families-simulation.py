"""Reproducible experiments for
'Exponential Families: The Natural Language of Modern Bandit Models'.

The script creates:
  - log_partition_map.pdf/png
  - sufficient_statistic_likelihood.pdf/png
  - conjugate_updates.pdf/png
  - rate_functions.pdf/png
  - exponential_family_regret.pdf/png
  - exponential_family_pull_counts.pdf/png
  - exponential_families_results.csv

All simulations use fixed random seeds.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import expit, gammaln
from scipy.stats import beta as beta_dist
from scipy.stats import gamma as gamma_dist
from scipy.stats import norm


OUT = Path(__file__).resolve().parent
SEED = 20260620


@dataclass(frozen=True)
class Config:
    horizon: int = 5000
    runs: int = 240
    binary_steps: int = 22

    bernoulli_means: tuple[float, ...] = (0.18, 0.24, 0.31, 0.36)
    gaussian_means: tuple[float, ...] = (0.00, 0.10, 0.18, 0.25)
    gaussian_sigma: float = 0.35
    poisson_means: tuple[float, ...] = (1.70, 2.00, 2.30, 2.60)


def bernoulli_kl(p: np.ndarray | float, q: np.ndarray | float) -> np.ndarray:
    p_arr = np.asarray(p, dtype=float)
    q_arr = np.asarray(q, dtype=float)
    q_safe = np.clip(q_arr, 1e-14, 1.0 - 1e-14)
    p_safe = np.clip(p_arr, 1e-14, 1.0 - 1e-14)
    term1 = np.where(p_arr > 0.0, p_safe * np.log(p_safe / q_safe), 0.0)
    term2 = np.where(
        p_arr < 1.0,
        (1.0 - p_safe) * np.log((1.0 - p_safe) / (1.0 - q_safe)),
        0.0,
    )
    return term1 + term2


def poisson_kl(p: np.ndarray | float, q: np.ndarray | float) -> np.ndarray:
    p_arr = np.asarray(p, dtype=float)
    q_arr = np.asarray(q, dtype=float)
    q_safe = np.clip(q_arr, 1e-14, None)
    p_safe = np.clip(p_arr, 1e-14, None)
    return np.where(p_arr > 0.0, p_safe * np.log(p_safe / q_safe), 0.0) + q_safe - p_arr


def bernoulli_kl_upper(emp: np.ndarray, counts: np.ndarray, beta: float, steps: int) -> np.ndarray:
    lo = emp.copy()
    hi = np.ones_like(emp)
    for _ in range(steps):
        mid = (lo + hi) / 2.0
        feasible = counts * bernoulli_kl(emp, mid) <= beta
        lo = np.where(feasible, mid, lo)
        hi = np.where(feasible, hi, mid)
    return lo


def poisson_kl_upper(emp: np.ndarray, counts: np.ndarray, beta: float, steps: int) -> np.ndarray:
    radius = beta / counts
    lo = np.maximum(emp, 0.0)
    hi = np.maximum(1.0, emp + 2.0 * np.sqrt(2.0 * (emp + 1.0) * radius) + 4.0 * radius + 2.0)
    # The bracket above is deliberately generous for the means used here.
    for _ in range(steps):
        mid = (lo + hi) / 2.0
        feasible = counts * poisson_kl(emp, mid) <= beta
        lo = np.where(feasible, mid, lo)
        hi = np.where(feasible, hi, mid)
    return lo


def plot_log_partition_map() -> pd.DataFrame:
    eta_b = np.linspace(-5.0, 5.0, 500)
    A_b = np.logaddexp(0.0, eta_b)
    m_b = expit(eta_b)
    v_b = m_b * (1.0 - m_b)

    eta_g = np.linspace(-3.0, 3.0, 500)
    A_g = 0.5 * eta_g**2
    m_g = eta_g
    v_g = np.ones_like(eta_g)

    eta_p = np.linspace(-2.5, 1.7, 500)
    A_p = np.exp(eta_p)
    m_p = np.exp(eta_p)
    v_p = np.exp(eta_p)

    fig, axes = plt.subplots(3, 3, figsize=(10.0, 8.0))
    families = [
        ("Bernoulli", eta_b, A_b, m_b, v_b),
        ("Gaussian, $\\sigma^2=1$", eta_g, A_g, m_g, v_g),
        ("Poisson", eta_p, A_p, m_p, v_p),
    ]
    for row, (name, eta, A, mean, var) in enumerate(families):
        axes[row, 0].plot(eta, A, linewidth=1.9)
        axes[row, 0].set_ylabel(name)
        axes[row, 1].plot(eta, mean, linewidth=1.9)
        axes[row, 2].plot(eta, var, linewidth=1.9)
        for col in range(3):
            axes[row, col].axhline(0.0, linewidth=0.6, alpha=0.45)
            axes[row, col].grid(True, linewidth=0.3, alpha=0.25)
            if row == 2:
                axes[row, col].set_xlabel(r"natural parameter $\eta$")
    axes[0, 0].set_title(r"$A(\eta)$")
    axes[0, 1].set_title(r"$A'(\eta)=\mathbb{E}[T(X)]$")
    axes[0, 2].set_title(r"$A''(\eta)=\mathrm{Var}(T(X))$")
    fig.suptitle("One function stores normalization, mean, and variance", y=1.01)
    fig.tight_layout()
    fig.savefig(OUT / "log_partition_map.pdf", bbox_inches="tight")
    fig.savefig(OUT / "log_partition_map.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    return pd.DataFrame(
        [
            {"experiment": "log_partition", "quantity": "Bernoulli mean at eta=0", "value": 0.5},
            {"experiment": "log_partition", "quantity": "Bernoulli variance at eta=0", "value": 0.25},
            {"experiment": "log_partition", "quantity": "Gaussian variance for sigma2=1", "value": 1.0},
            {"experiment": "log_partition", "quantity": "Poisson mean at eta=log(2.5)", "value": 2.5},
        ]
    )


def plot_sufficient_statistic_likelihood() -> pd.DataFrame:
    n = 12
    sequences = {
        "sequence A: 7 clicks": np.array([1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1]),
        "sequence B: same 7 clicks": np.array([0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1]),
        "sequence C: 4 clicks": np.array([1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0]),
    }
    p = np.linspace(0.01, 0.99, 600)

    fig, ax = plt.subplots(figsize=(7.3, 4.55))
    rows: list[dict[str, float | str]] = []
    styles = ["-", "--", "-."]
    for (label, seq), style in zip(sequences.items(), styles):
        s = int(seq.sum())
        log_lik = s * np.log(p) + (n - s) * np.log(1.0 - p)
        log_lik -= log_lik.max()
        ax.plot(p, np.exp(log_lik), linestyle=style, linewidth=2.0, label=label)
        rows.append({"experiment": "sufficiency", "quantity": f"success count for {label}", "value": float(s)})
    ax.set_xlabel(r"candidate click probability $p$")
    ax.set_ylabel("relative likelihood")
    ax.set_title("Order disappears: the likelihood remembers only the success count")
    ax.legend(frameon=False)
    ax.grid(True, linewidth=0.3, alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "sufficient_statistic_likelihood.pdf", bbox_inches="tight")
    fig.savefig(OUT / "sufficient_statistic_likelihood.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    return pd.DataFrame(rows)


def plot_conjugate_updates() -> pd.DataFrame:
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.55))
    rows: list[dict[str, float | str]] = []

    # Beta-Bernoulli
    p = np.linspace(0.001, 0.999, 600)
    a0, b0 = 1.0, 1.0
    successes, failures = 14, 6
    a1, b1 = a0 + successes, b0 + failures
    axes[0].plot(p, beta_dist.pdf(p, a0, b0), linewidth=1.7, label="prior")
    axes[0].plot(p, beta_dist.pdf(p, a1, b1), linewidth=2.0, label="posterior")
    axes[0].axvline(successes / (successes + failures), linewidth=0.9, linestyle="--")
    axes[0].set_title("Bernoulli mean")
    axes[0].set_xlabel(r"$p$")
    axes[0].set_ylabel("density")
    axes[0].legend(frameon=False)
    rows.append({"experiment": "conjugacy", "quantity": "Beta posterior mean", "value": a1 / (a1 + b1)})

    # Gaussian known variance
    mu = np.linspace(-0.8, 1.2, 600)
    prior_mean, prior_sd = 0.0, 0.7
    sigma = 0.5
    n, xbar = 12, 0.45
    prior_precision = 1.0 / prior_sd**2
    data_precision = n / sigma**2
    post_var = 1.0 / (prior_precision + data_precision)
    post_mean = post_var * (prior_precision * prior_mean + data_precision * xbar)
    axes[1].plot(mu, norm.pdf(mu, prior_mean, prior_sd), linewidth=1.7, label="prior")
    axes[1].plot(mu, norm.pdf(mu, post_mean, math.sqrt(post_var)), linewidth=2.0, label="posterior")
    axes[1].axvline(xbar, linewidth=0.9, linestyle="--")
    axes[1].set_title("Gaussian mean")
    axes[1].set_xlabel(r"$\mu$")
    axes[1].legend(frameon=False)
    rows.append({"experiment": "conjugacy", "quantity": "Gaussian posterior mean", "value": post_mean})

    # Gamma-Poisson (shape-rate)
    lam = np.linspace(0.001, 5.5, 600)
    shape0, rate0 = 1.0, 1.0
    n_pois, total = 12, 30
    shape1, rate1 = shape0 + total, rate0 + n_pois
    axes[2].plot(lam, gamma_dist.pdf(lam, a=shape0, scale=1.0 / rate0), linewidth=1.7, label="prior")
    axes[2].plot(lam, gamma_dist.pdf(lam, a=shape1, scale=1.0 / rate1), linewidth=2.0, label="posterior")
    axes[2].axvline(total / n_pois, linewidth=0.9, linestyle="--")
    axes[2].set_title("Poisson rate")
    axes[2].set_xlabel(r"$\lambda$")
    axes[2].legend(frameon=False)
    rows.append({"experiment": "conjugacy", "quantity": "Gamma posterior mean", "value": shape1 / rate1})

    for ax in axes:
        ax.grid(True, linewidth=0.3, alpha=0.25)
    fig.suptitle("Conjugacy is a bookkeeping rule: old pseudo-data plus new data", y=1.03)
    fig.tight_layout()
    fig.savefig(OUT / "conjugate_updates.pdf", bbox_inches="tight")
    fig.savefig(OUT / "conjugate_updates.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    return pd.DataFrame(rows)


def plot_rate_functions() -> pd.DataFrame:
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.55))
    rows: list[dict[str, float | str]] = []

    mu_b = 0.35
    x_b = np.linspace(0.005, 0.995, 600)
    rate_b = bernoulli_kl(x_b, mu_b)
    axes[0].plot(x_b, rate_b, linewidth=2.0)
    axes[0].axvline(mu_b, linewidth=0.9, linestyle="--")
    axes[0].set_title("Bernoulli")
    axes[0].set_xlabel(r"sample mean $x$")
    axes[0].set_ylabel("rate function")

    mu_g, sigma = 0.25, 0.35
    x_g = np.linspace(-0.8, 1.3, 600)
    rate_g = (x_g - mu_g) ** 2 / (2.0 * sigma**2)
    axes[1].plot(x_g, rate_g, linewidth=2.0)
    axes[1].axvline(mu_g, linewidth=0.9, linestyle="--")
    axes[1].set_title("Gaussian")
    axes[1].set_xlabel(r"sample mean $x$")

    mu_p = 2.6
    x_p = np.linspace(0.01, 6.5, 600)
    rate_p = poisson_kl(x_p, mu_p)
    axes[2].plot(x_p, rate_p, linewidth=2.0)
    axes[2].axvline(mu_p, linewidth=0.9, linestyle="--")
    axes[2].set_title("Poisson")
    axes[2].set_xlabel(r"sample mean $x$")

    for ax in axes:
        ax.set_ylim(bottom=0.0)
        ax.grid(True, linewidth=0.3, alpha=0.25)
    fig.suptitle("Different reward models, the same concentration story", y=1.03)
    fig.tight_layout()
    fig.savefig(OUT / "rate_functions.pdf", bbox_inches="tight")
    fig.savefig(OUT / "rate_functions.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    rows.extend(
        [
            {"experiment": "rate_function", "quantity": "Bernoulli rate at x=0.45", "value": float(bernoulli_kl(0.45, mu_b))},
            {"experiment": "rate_function", "quantity": "Gaussian rate at x=0.45", "value": float((0.45 - mu_g) ** 2 / (2.0 * sigma**2))},
            {"experiment": "rate_function", "quantity": "Poisson rate at x=3.2", "value": float(poisson_kl(3.2, mu_p))},
        ]
    )
    return pd.DataFrame(rows)


def _beta_time(t: int) -> float:
    if t <= 2:
        return 1.0
    return math.log(t) + 3.0 * math.log(max(math.log(t), 1.0))


def simulate_bernoulli(cfg: Config, algorithm: str, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    means = np.asarray(cfg.bernoulli_means, dtype=float)
    runs, horizon, k = cfg.runs, cfg.horizon, means.size
    counts = np.zeros((runs, k), dtype=float)
    sums = np.zeros((runs, k), dtype=float)
    regrets = np.zeros((runs, horizon), dtype=float)
    best = float(means.max())
    rr = np.arange(runs)

    for a in range(k):
        reward = (rng.random(runs) < means[a]).astype(float)
        counts[:, a] += 1.0
        sums[:, a] += reward
        regrets[:, a] = (best - means[a]) + (regrets[:, a - 1] if a > 0 else 0.0)

    for t in range(k, horizon):
        emp = sums / counts
        if algorithm == "KL-UCB":
            index = bernoulli_kl_upper(emp, counts, _beta_time(t + 1), cfg.binary_steps)
        elif algorithm == "Thompson sampling":
            index = rng.beta(1.0 + sums, 1.0 + counts - sums)
        else:
            raise ValueError(f"unknown algorithm: {algorithm}")
        action = np.argmax(index, axis=1)
        reward = (rng.random(runs) < means[action]).astype(float)
        counts[rr, action] += 1.0
        sums[rr, action] += reward
        regrets[:, t] = regrets[:, t - 1] + best - means[action]
    return regrets, counts


def simulate_gaussian(cfg: Config, algorithm: str, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    means = np.asarray(cfg.gaussian_means, dtype=float)
    sigma = cfg.gaussian_sigma
    runs, horizon, k = cfg.runs, cfg.horizon, means.size
    counts = np.zeros((runs, k), dtype=float)
    sums = np.zeros((runs, k), dtype=float)
    regrets = np.zeros((runs, horizon), dtype=float)
    best = float(means.max())
    rr = np.arange(runs)

    for a in range(k):
        reward = rng.normal(means[a], sigma, size=runs)
        counts[:, a] += 1.0
        sums[:, a] += reward
        regrets[:, a] = (best - means[a]) + (regrets[:, a - 1] if a > 0 else 0.0)

    prior_mean, prior_var = 0.0, 1.0
    for t in range(k, horizon):
        emp = sums / counts
        if algorithm == "KL-UCB":
            index = emp + np.sqrt(2.0 * sigma**2 * _beta_time(t + 1) / counts)
        elif algorithm == "Thompson sampling":
            precision = 1.0 / prior_var + counts / sigma**2
            post_var = 1.0 / precision
            post_mean = post_var * (prior_mean / prior_var + sums / sigma**2)
            index = rng.normal(post_mean, np.sqrt(post_var))
        else:
            raise ValueError(f"unknown algorithm: {algorithm}")
        action = np.argmax(index, axis=1)
        reward = rng.normal(means[action], sigma)
        counts[rr, action] += 1.0
        sums[rr, action] += reward
        regrets[:, t] = regrets[:, t - 1] + best - means[action]
    return regrets, counts


def simulate_poisson(cfg: Config, algorithm: str, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    means = np.asarray(cfg.poisson_means, dtype=float)
    runs, horizon, k = cfg.runs, cfg.horizon, means.size
    counts = np.zeros((runs, k), dtype=float)
    sums = np.zeros((runs, k), dtype=float)
    regrets = np.zeros((runs, horizon), dtype=float)
    best = float(means.max())
    rr = np.arange(runs)

    for a in range(k):
        reward = rng.poisson(means[a], size=runs).astype(float)
        counts[:, a] += 1.0
        sums[:, a] += reward
        regrets[:, a] = (best - means[a]) + (regrets[:, a - 1] if a > 0 else 0.0)

    for t in range(k, horizon):
        emp = sums / counts
        if algorithm == "KL-UCB":
            index = poisson_kl_upper(emp, counts, _beta_time(t + 1), cfg.binary_steps)
        elif algorithm == "Thompson sampling":
            index = rng.gamma(shape=1.0 + sums, scale=1.0 / (1.0 + counts))
        else:
            raise ValueError(f"unknown algorithm: {algorithm}")
        action = np.argmax(index, axis=1)
        reward = rng.poisson(means[action]).astype(float)
        counts[rr, action] += 1.0
        sums[rr, action] += reward
        regrets[:, t] = regrets[:, t - 1] + best - means[action]
    return regrets, counts


def run_bandit_experiments(cfg: Config) -> pd.DataFrame:
    families = {
        "Bernoulli": (simulate_bernoulli, np.asarray(cfg.bernoulli_means)),
        "Gaussian": (simulate_gaussian, np.asarray(cfg.gaussian_means)),
        "Poisson": (simulate_poisson, np.asarray(cfg.poisson_means)),
    }
    algorithms = ("KL-UCB", "Thompson sampling")
    curves: dict[tuple[str, str], np.ndarray] = {}
    counts_map: dict[tuple[str, str], np.ndarray] = {}
    rows: list[dict[str, float | str]] = []

    for f_idx, (family, (simulator, means)) in enumerate(families.items()):
        for a_idx, algorithm in enumerate(algorithms):
            regrets, counts = simulator(cfg, algorithm, SEED + 1000 * f_idx + 100 * a_idx)
            curves[(family, algorithm)] = regrets.mean(axis=0)
            counts_map[(family, algorithm)] = counts.mean(axis=0)
            final = regrets[:, -1]
            rows.extend(
                [
                    {"experiment": "bandit", "family": family, "algorithm": algorithm, "quantity": "mean final pseudo-regret", "value": float(final.mean())},
                    {"experiment": "bandit", "family": family, "algorithm": algorithm, "quantity": "standard error final pseudo-regret", "value": float(final.std(ddof=1) / math.sqrt(cfg.runs))},
                    {"experiment": "bandit", "family": family, "algorithm": algorithm, "quantity": "mean pulls of best arm", "value": float(counts[:, int(np.argmax(means))].mean())},
                ]
            )

    time = np.arange(1, cfg.horizon + 1)
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.55), sharex=True)
    for ax, family in zip(axes, families.keys()):
        for algorithm in algorithms:
            ax.plot(time, curves[(family, algorithm)], linewidth=1.9, label=algorithm)
        ax.set_title(family)
        ax.set_xlabel("round")
        ax.grid(True, linewidth=0.3, alpha=0.25)
    axes[0].set_ylabel("mean pseudo-regret")
    axes[-1].legend(frameon=False)
    fig.suptitle("The policy skeleton stays the same; the family supplies the right evidence scale", y=1.03)
    fig.tight_layout()
    fig.savefig(OUT / "exponential_family_regret.pdf", bbox_inches="tight")
    fig.savefig(OUT / "exponential_family_regret.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.55))
    x = np.arange(4)
    width = 0.36
    for ax, (family, (_, means)) in zip(axes, families.items()):
        for j, algorithm in enumerate(algorithms):
            ax.bar(x + (j - 0.5) * width, counts_map[(family, algorithm)], width=width, label=algorithm)
        ax.set_title(family)
        ax.set_xticks(x, [f"arm {j+1}" for j in x])
        ax.set_xlabel("arm")
        ax.grid(True, axis="y", linewidth=0.3, alpha=0.25)
    axes[0].set_ylabel("mean number of pulls")
    axes[-1].legend(frameon=False)
    fig.suptitle("Most samples eventually flow to the best mean", y=1.03)
    fig.tight_layout()
    fig.savefig(OUT / "exponential_family_pull_counts.pdf", bbox_inches="tight")
    fig.savefig(OUT / "exponential_family_pull_counts.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    return pd.DataFrame(rows)


def main() -> None:
    cfg = Config()
    frames = [
        plot_log_partition_map(),
        plot_sufficient_statistic_likelihood(),
        plot_conjugate_updates(),
        plot_rate_functions(),
        run_bandit_experiments(cfg),
    ]
    results = pd.concat(frames, ignore_index=True, sort=False)
    results.to_csv(OUT / "exponential_families_results.csv", index=False)
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
