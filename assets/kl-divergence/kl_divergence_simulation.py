"""Reproducible experiments for
'KL Divergence as the Geometry of Statistical Evidence'.

The script creates:
  - bernoulli_kl_geometry.pdf/png
  - evidence_accumulation.pdf/png
  - tail_bounds_comparison.pdf/png
  - confidence_radii.pdf/png
  - kl_ucb_regret.pdf/png
  - kl_ucb_pull_counts.pdf/png
  - kl_divergence_results.csv

All simulations use fixed random seeds.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binom


OUT = Path(__file__).resolve().parent
SEED = 20260619


@dataclass(frozen=True)
class Config:
    evidence_p: float = 0.62
    evidence_q: float = 0.50
    evidence_horizon: int = 220
    evidence_paths: int = 14

    tail_p: float = 0.20
    tail_n: int = 100

    confidence_n: int = 120
    confidence_delta: float = 0.05

    bandit_means: tuple[float, ...] = (0.03, 0.05, 0.08, 0.12)
    bandit_horizon: int = 12000
    bandit_runs: int = 220
    kl_binary_steps: int = 19


def bernoulli_kl(p: np.ndarray | float, q: np.ndarray | float) -> np.ndarray:
    """Binary relative entropy d(p || q), with stable boundary handling."""
    p_arr = np.asarray(p, dtype=float)
    q_arr = np.asarray(q, dtype=float)
    q_safe = np.clip(q_arr, 1e-14, 1.0 - 1e-14)

    p_safe = np.clip(p_arr, 1e-14, 1.0 - 1e-14)
    first_raw = p_safe * np.log(p_safe / q_safe)
    second_raw = (1.0 - p_safe) * np.log((1.0 - p_safe) / (1.0 - q_safe))
    first = np.where(p_arr > 0.0, first_raw, 0.0)
    second = np.where(p_arr < 1.0, second_raw, 0.0)
    return first + second


def kl_upper_bound(
    empirical_mean: np.ndarray,
    counts: np.ndarray,
    beta: float,
    steps: int = 28,
) -> np.ndarray:
    """Largest q in [empirical_mean, 1] with N d(mean || q) <= beta."""
    lo = empirical_mean.copy()
    hi = np.ones_like(empirical_mean)
    for _ in range(steps):
        mid = (lo + hi) / 2.0
        feasible = counts * bernoulli_kl(empirical_mean, mid) <= beta
        lo = np.where(feasible, mid, lo)
        hi = np.where(feasible, hi, mid)
    return lo


def plot_bernoulli_kl_geometry() -> pd.DataFrame:
    q = np.linspace(0.002, 0.998, 800)
    reference_ps = (0.10, 0.50, 0.90)

    fig, ax = plt.subplots(figsize=(7.2, 4.45))
    for p in reference_ps:
        ax.plot(q, bernoulli_kl(p, q), linewidth=1.9, label=fr"$p={p:.1f}$")
        ax.scatter([p], [0.0], s=24)
    ax.set_xlabel("candidate mean $q$")
    ax.set_ylabel(r"Bernoulli KL divergence $d(p\Vert q)$")
    ax.set_ylim(0.0, 2.8)
    ax.set_title("The same numerical error carries different evidence near a boundary")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "bernoulli_kl_geometry.pdf", bbox_inches="tight")
    fig.savefig(OUT / "bernoulli_kl_geometry.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    rows = []
    for p in reference_ps:
        for offset in (-0.05, 0.05):
            candidate = p + offset
            if 0.0 < candidate < 1.0:
                rows.append(
                    {
                        "experiment": "geometry",
                        "quantity": f"d({p:.2f}||{candidate:.2f})",
                        "value": float(bernoulli_kl(p, candidate)),
                    }
                )
    return pd.DataFrame(rows)


def plot_evidence_accumulation(rng: np.random.Generator, cfg: Config) -> pd.DataFrame:
    p = cfg.evidence_p
    q = cfg.evidence_q
    n = cfg.evidence_horizon
    x = rng.random((cfg.evidence_paths, n)) < p
    increments = np.where(x, math.log(p / q), math.log((1.0 - p) / (1.0 - q)))
    paths = np.cumsum(increments, axis=1)
    time = np.arange(1, n + 1)
    expected = time * float(bernoulli_kl(p, q))

    fig, ax = plt.subplots(figsize=(7.2, 4.55))
    for path in paths:
        ax.plot(time, path, linewidth=0.95, alpha=0.62)
    ax.plot(time, expected, linewidth=2.4, label=r"expected drift $n\,d(p\Vert q)$")
    ax.axhline(0.0, linewidth=0.85)
    ax.set_xlabel("number of observations $n$")
    ax.set_ylabel(r"cumulative log-evidence $\log(P/Q)$")
    ax.set_title("Evidence is noisy path by path, but its average drift is KL")
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "evidence_accumulation.pdf", bbox_inches="tight")
    fig.savefig(OUT / "evidence_accumulation.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    audit_x = rng.random((20000, n)) < p
    audit_increments = np.where(
        audit_x,
        math.log(p / q),
        math.log((1.0 - p) / (1.0 - q)),
    )
    audit_final = audit_increments.sum(axis=1)

    return pd.DataFrame(
        [
            {
                "experiment": "evidence",
                "quantity": "KL per observation",
                "value": float(bernoulli_kl(p, q)),
            },
            {
                "experiment": "evidence",
                "quantity": "theoretical expected final log evidence",
                "value": float(expected[-1]),
            },
            {
                "experiment": "evidence",
                "quantity": "simulated mean final log evidence (20000 paths)",
                "value": float(audit_final.mean()),
            },
        ]
    )


def plot_tail_bounds(cfg: Config) -> pd.DataFrame:
    p = cfg.tail_p
    n = cfg.tail_n
    counts = np.arange(math.floor(n * p) + 1, math.floor(0.56 * n) + 1)
    x = counts / n

    exact = binom.sf(counts - 1, n, p)
    kl_bound = np.exp(-n * bernoulli_kl(x, p))
    hoeffding = np.exp(-2.0 * n * (x - p) ** 2)

    fig, ax = plt.subplots(figsize=(7.2, 4.45))
    ax.semilogy(x, exact, marker="o", markersize=3.5, linewidth=1.4, label="exact binomial tail")
    ax.semilogy(x, kl_bound, linewidth=1.9, label=r"KL-Chernoff: $e^{-n d(x\Vert p)}$")
    ax.semilogy(x, hoeffding, linewidth=1.7, linestyle="--", label=r"Hoeffding: $e^{-2n(x-p)^2}$")
    ax.set_xlabel(r"threshold $x$ in $\Pr(\widehat p_n\geq x)$")
    ax.set_ylabel("upper-tail probability (log scale)")
    ax.set_title("KL keeps the Bernoulli shape that Hoeffding discards")
    ax.legend(frameon=False)
    ax.grid(True, which="both", linewidth=0.35, alpha=0.35)
    fig.tight_layout()
    fig.savefig(OUT / "tail_bounds_comparison.pdf", bbox_inches="tight")
    fig.savefig(OUT / "tail_bounds_comparison.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    selected = [0, len(x) // 3, 2 * len(x) // 3, len(x) - 1]
    rows = []
    for j in selected:
        rows.extend(
            [
                {
                    "experiment": "tail_bound",
                    "quantity": f"exact tail at x={x[j]:.2f}",
                    "value": float(exact[j]),
                },
                {
                    "experiment": "tail_bound",
                    "quantity": f"KL bound at x={x[j]:.2f}",
                    "value": float(kl_bound[j]),
                },
                {
                    "experiment": "tail_bound",
                    "quantity": f"Hoeffding bound at x={x[j]:.2f}",
                    "value": float(hoeffding[j]),
                },
            ]
        )
    return pd.DataFrame(rows)


def plot_confidence_radii(cfg: Config) -> pd.DataFrame:
    empirical = np.linspace(0.001, 0.999, 500)
    counts = np.full_like(empirical, cfg.confidence_n, dtype=float)
    beta = math.log(1.0 / cfg.confidence_delta)
    kl_upper = kl_upper_bound(empirical, counts, beta, steps=32)
    kl_radius = kl_upper - empirical
    hoeffding_radius = np.minimum(
        1.0 - empirical,
        math.sqrt(beta / (2.0 * cfg.confidence_n)) * np.ones_like(empirical),
    )

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(empirical, kl_radius, linewidth=2.0, label="KL upper radius")
    ax.plot(empirical, hoeffding_radius, linewidth=1.8, linestyle="--", label="Hoeffding upper radius")
    ax.set_xlabel(r"empirical Bernoulli mean $\widehat p$")
    ax.set_ylabel("distance from estimate to upper confidence endpoint")
    ax.set_title(fr"Evidence-shaped confidence bounds ($n={cfg.confidence_n}$, $\delta={cfg.confidence_delta}$)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "confidence_radii.pdf", bbox_inches="tight")
    fig.savefig(OUT / "confidence_radii.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    rows = []
    for p_hat in (0.02, 0.10, 0.50, 0.90):
        j = int(np.argmin(np.abs(empirical - p_hat)))
        rows.extend(
            [
                {
                    "experiment": "confidence_radius",
                    "quantity": f"KL radius at mean={p_hat:.2f}",
                    "value": float(kl_radius[j]),
                },
                {
                    "experiment": "confidence_radius",
                    "quantity": f"Hoeffding radius at mean={p_hat:.2f}",
                    "value": float(hoeffding_radius[j]),
                },
            ]
        )
    return pd.DataFrame(rows)


def _initialize_bandit(
    rng: np.random.Generator,
    means: np.ndarray,
    runs: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    k = len(means)
    successes = np.zeros((runs, k), dtype=np.int32)
    counts = np.zeros((runs, k), dtype=np.int32)
    cumulative_regret = np.zeros(runs, dtype=float)
    curve = np.zeros(1, dtype=float)

    for arm in range(k):
        reward = rng.random(runs) < means[arm]
        successes[:, arm] += reward
        counts[:, arm] += 1
        cumulative_regret += means.max() - means[arm]

    return successes, counts, cumulative_regret, curve


def simulate_ucb(
    rng: np.random.Generator,
    means: np.ndarray,
    horizon: int,
    runs: int,
) -> tuple[np.ndarray, np.ndarray]:
    k = len(means)
    successes, counts, cumulative_regret, _ = _initialize_bandit(rng, means, runs)
    curve = np.zeros(horizon + 1, dtype=float)

    running = 0.0
    for arm in range(k):
        running += means.max() - means[arm]
        curve[arm + 1] = running
    curve[k] = cumulative_regret.mean()

    row = np.arange(runs)
    for t in range(k, horizon):
        empirical = successes / counts
        index = empirical + np.sqrt(2.0 * math.log(t + 1.0) / counts)
        chosen = np.argmax(index, axis=1)
        reward = rng.random(runs) < means[chosen]
        successes[row, chosen] += reward
        counts[row, chosen] += 1
        cumulative_regret += means.max() - means[chosen]
        curve[t + 1] = cumulative_regret.mean()
    return curve, counts.mean(axis=0)


def simulate_kl_ucb(
    rng: np.random.Generator,
    means: np.ndarray,
    horizon: int,
    runs: int,
    binary_steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    k = len(means)
    successes, counts, cumulative_regret, _ = _initialize_bandit(rng, means, runs)
    curve = np.zeros(horizon + 1, dtype=float)

    running = 0.0
    for arm in range(k):
        running += means.max() - means[arm]
        curve[arm + 1] = running
    curve[k] = cumulative_regret.mean()

    row = np.arange(runs)
    for t in range(k, horizon):
        empirical = successes / counts
        log_t = math.log(t + 1.0)
        beta = log_t + 3.0 * math.log(max(log_t, 1.0))
        index = kl_upper_bound(empirical, counts.astype(float), beta, steps=binary_steps)
        chosen = np.argmax(index, axis=1)
        reward = rng.random(runs) < means[chosen]
        successes[row, chosen] += reward
        counts[row, chosen] += 1
        cumulative_regret += means.max() - means[chosen]
        curve[t + 1] = cumulative_regret.mean()
    return curve, counts.mean(axis=0)


def simulate_thompson(
    rng: np.random.Generator,
    means: np.ndarray,
    horizon: int,
    runs: int,
) -> tuple[np.ndarray, np.ndarray]:
    k = len(means)
    successes, counts, cumulative_regret, _ = _initialize_bandit(rng, means, runs)
    curve = np.zeros(horizon + 1, dtype=float)

    running = 0.0
    for arm in range(k):
        running += means.max() - means[arm]
        curve[arm + 1] = running
    curve[k] = cumulative_regret.mean()

    row = np.arange(runs)
    for t in range(k, horizon):
        samples = rng.beta(successes + 1.0, counts - successes + 1.0)
        chosen = np.argmax(samples, axis=1)
        reward = rng.random(runs) < means[chosen]
        successes[row, chosen] += reward
        counts[row, chosen] += 1
        cumulative_regret += means.max() - means[chosen]
        curve[t + 1] = cumulative_regret.mean()
    return curve, counts.mean(axis=0)


def plot_bandit_experiment(cfg: Config) -> pd.DataFrame:
    means = np.asarray(cfg.bandit_means, dtype=float)

    ucb_curve, ucb_counts = simulate_ucb(
        np.random.default_rng(SEED + 101), means, cfg.bandit_horizon, cfg.bandit_runs
    )
    kl_curve, kl_counts = simulate_kl_ucb(
        np.random.default_rng(SEED + 202),
        means,
        cfg.bandit_horizon,
        cfg.bandit_runs,
        cfg.kl_binary_steps,
    )
    ts_curve, ts_counts = simulate_thompson(
        np.random.default_rng(SEED + 303), means, cfg.bandit_horizon, cfg.bandit_runs
    )

    time = np.arange(cfg.bandit_horizon + 1)
    fig, ax = plt.subplots(figsize=(7.25, 4.55))
    ax.plot(time, ucb_curve, linewidth=1.8, label="UCB1")
    ax.plot(time, kl_curve, linewidth=2.0, label="KL-UCB")
    ax.plot(time, ts_curve, linewidth=1.8, label="Thompson sampling")
    ax.set_xlabel("round $t$")
    ax.set_ylabel("mean cumulative pseudo-regret")
    ax.set_title("A Bernoulli-aware confidence bound spends evidence more carefully")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "kl_ucb_regret.pdf", bbox_inches="tight")
    fig.savefig(OUT / "kl_ucb_regret.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    labels = [f"arm {i + 1}\n$\\mu={m:.2f}$" for i, m in enumerate(means)]
    x = np.arange(len(means))
    width = 0.25
    fig, ax = plt.subplots(figsize=(7.25, 4.5))
    ax.bar(x - width, ucb_counts, width, label="UCB1")
    ax.bar(x, kl_counts, width, label="KL-UCB")
    ax.bar(x + width, ts_counts, width, label="Thompson sampling")
    ax.set_xticks(x, labels)
    ax.set_ylabel("mean number of pulls")
    ax.set_title(fr"Where the algorithms spent {cfg.bandit_horizon:,} observations")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "kl_ucb_pull_counts.pdf", bbox_inches="tight")
    fig.savefig(OUT / "kl_ucb_pull_counts.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    rows = []
    for name, curve, counts in (
        ("UCB1", ucb_curve, ucb_counts),
        ("KL-UCB", kl_curve, kl_counts),
        ("Thompson sampling", ts_curve, ts_counts),
    ):
        rows.append(
            {
                "experiment": "bandit",
                "quantity": f"final regret: {name}",
                "value": float(curve[-1]),
            }
        )
        for arm, count in enumerate(counts, start=1):
            rows.append(
                {
                    "experiment": "bandit",
                    "quantity": f"mean pulls arm {arm}: {name}",
                    "value": float(count),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    cfg = Config()
    rng = np.random.default_rng(SEED)

    frames = [
        plot_bernoulli_kl_geometry(),
        plot_evidence_accumulation(rng, cfg),
        plot_tail_bounds(cfg),
        plot_confidence_radii(cfg),
        plot_bandit_experiment(cfg),
    ]
    results = pd.concat(frames, ignore_index=True)
    results.to_csv(OUT / "kl_divergence_results.csv", index=False)

    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
