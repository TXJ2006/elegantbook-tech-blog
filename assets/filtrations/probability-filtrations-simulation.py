"""
Simulation for the lecture note:
Probability, Filtrations, and Adaptive Data.

The goal is not to build a large benchmark.  The goal is to make three
probability ideas visible:

1. A sample average is reliable at one fixed time.
2. Looking many times with the same fixed-time error bar creates false discoveries.
3. Selecting the largest empirical mean among many arms creates upward bias.

Run:
    python probability-filtrations-simulation.py
"""
from __future__ import annotations

import os
from dataclasses import dataclass
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

OUT = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(OUT, "figures")
os.makedirs(FIG, exist_ok=True)

rng = np.random.default_rng(7)


@dataclass
class Config:
    p: float = 0.50
    runs: int = 20000
    fixed_n: int = 200
    stop_start: int = 20
    threshold: float = 0.56
    alpha: float = 0.05
    k_arms: int = 20
    n_per_arm: int = 20


def hoeffding_radius(n: np.ndarray | float, alpha: float) -> np.ndarray | float:
    """Fixed-time two-sided Hoeffding radius for Bernoulli rewards in [0,1]."""
    return np.sqrt(np.log(2.0 / alpha) / (2.0 * n))


def union_radius(n: np.ndarray | float, alpha: float, T: int) -> np.ndarray | float:
    """A simple anytime radius obtained by a union bound over t=1,...,T."""
    return np.sqrt(np.log(2.0 * T / alpha) / (2.0 * n))


def simulate_fixed_and_stopped(cfg: Config):
    fixed_means = np.empty(cfg.runs)
    stopped_means = np.empty(cfg.runs)
    stopped_times = np.empty(cfg.runs, dtype=int)
    stopped_by_rule = np.empty(cfg.runs, dtype=bool)
    final_normal_false_alarm = np.empty(cfg.runs, dtype=bool)
    anytime_normal_false_alarm = np.empty(cfg.runs, dtype=bool)
    anytime_union_false_alarm = np.empty(cfg.runs, dtype=bool)

    for r in range(cfg.runs):
        x = rng.binomial(1, cfg.p, size=cfg.fixed_n)
        s = np.cumsum(x)
        t = np.arange(1, cfg.fixed_n + 1)
        m = s / t

        fixed_means[r] = m[-1]

        normal_radius = 1.96 * np.sqrt(0.25 / t)
        union_r = union_radius(t, cfg.alpha, cfg.fixed_n)
        final_normal_false_alarm[r] = abs(m[-1] - cfg.p) > normal_radius[-1]
        anytime_normal_false_alarm[r] = np.any(np.abs(m - cfg.p) > normal_radius)
        anytime_union_false_alarm[r] = np.any(np.abs(m - cfg.p) > union_r)

        eligible = np.where((t >= cfg.stop_start) & (m >= cfg.threshold))[0]
        if eligible.size > 0:
            j = int(eligible[0])
            stopped_times[r] = j + 1
            stopped_means[r] = m[j]
            stopped_by_rule[r] = True
        else:
            stopped_times[r] = cfg.fixed_n
            stopped_means[r] = m[-1]
            stopped_by_rule[r] = False

    rad_fixed = hoeffding_radius(cfg.fixed_n, cfg.alpha)
    fixed_cover = np.mean(np.abs(fixed_means - cfg.p) <= rad_fixed)

    rad_stopped_naive = hoeffding_radius(stopped_times, cfg.alpha)
    stopped_cover_naive = np.mean(np.abs(stopped_means - cfg.p) <= rad_stopped_naive)

    rad_stopped_union = union_radius(stopped_times, cfg.alpha, cfg.fixed_n)
    stopped_cover_union = np.mean(np.abs(stopped_means - cfg.p) <= rad_stopped_union)

    return {
        "fixed_means": fixed_means,
        "stopped_means": stopped_means,
        "stopped_times": stopped_times,
        "stopped_by_rule": stopped_by_rule,
        "fixed_cover": fixed_cover,
        "stopped_cover_naive": stopped_cover_naive,
        "stopped_cover_union": stopped_cover_union,
        "final_normal_false_alarm": np.mean(final_normal_false_alarm),
        "anytime_normal_false_alarm": np.mean(anytime_normal_false_alarm),
        "anytime_union_false_alarm": np.mean(anytime_union_false_alarm),
        "stop_rate": np.mean(stopped_by_rule),
        "avg_stop_time": np.mean(stopped_times),
        "fixed_mean_avg": np.mean(fixed_means),
        "stopped_mean_avg": np.mean(stopped_means),
    }


def simulate_post_selection(cfg: Config):
    # All arms have the same true mean.  Any winner is a winner only because of noise.
    samples = rng.binomial(1, cfg.p, size=(cfg.runs, cfg.k_arms, cfg.n_per_arm))
    means = samples.mean(axis=2)
    selected = means.max(axis=1)
    ordinary = means[:, 0]
    return {
        "selected_means": selected,
        "ordinary_means": ordinary,
        "selected_mean_avg": float(np.mean(selected)),
        "ordinary_mean_avg": float(np.mean(ordinary)),
        "selection_bias": float(np.mean(selected) - cfg.p),
    }


def make_plots(cfg: Config, fs, ps):
    plt.figure(figsize=(6.6, 4.0))
    bins = np.linspace(0.35, 0.75, 41)
    plt.hist(fs["fixed_means"], bins=bins, alpha=0.6, density=True, label="fixed time average")
    plt.hist(fs["stopped_means"], bins=bins, alpha=0.6, density=True, label="average after adaptive stopping")
    plt.axvline(cfg.p, linewidth=2, label="true mean")
    plt.xlabel("observed sample mean")
    plt.ylabel("density")
    plt.title("Fixed sampling versus adaptive stopping")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "fixed_vs_stopped_means.pdf"))
    plt.savefig(os.path.join(FIG, "fixed_vs_stopped_means.png"), dpi=200)
    plt.close()

    labels = ["one final look\nnormal bar", "many peeks\nsame bar", "many peeks\nunion bar"]
    vals = [fs["final_normal_false_alarm"], fs["anytime_normal_false_alarm"], fs["anytime_union_false_alarm"]]
    plt.figure(figsize=(6.4, 3.8))
    plt.bar(labels, vals)
    plt.axhline(0.05, linestyle="--", linewidth=1.5, label="5% target")
    plt.ylim(0.0, 0.50)
    plt.ylabel("false alarm rate")
    plt.title("Peeking many times changes the meaning of an error bar")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "false_alarm_comparison.pdf"))
    plt.savefig(os.path.join(FIG, "false_alarm_comparison.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(6.6, 4.0))
    bins = np.linspace(0.2, 0.95, 41)
    plt.hist(ps["ordinary_means"], bins=bins, alpha=0.6, density=True, label="one fixed arm")
    plt.hist(ps["selected_means"], bins=bins, alpha=0.6, density=True, label="best empirical arm among 20")
    plt.axvline(cfg.p, linewidth=2, label="true mean")
    plt.xlabel("empirical mean")
    plt.ylabel("density")
    plt.title("Post-selection optimism when all arms are equal")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "post_selection_bias.pdf"))
    plt.savefig(os.path.join(FIG, "post_selection_bias.png"), dpi=200)
    plt.close()


def main():
    cfg = Config()
    fs = simulate_fixed_and_stopped(cfg)
    ps = simulate_post_selection(cfg)
    make_plots(cfg, fs, ps)

    rows = [
        {"quantity": "true Bernoulli mean", "value": cfg.p},
        {"quantity": "runs", "value": cfg.runs},
        {"quantity": "fixed sample size", "value": cfg.fixed_n},
        {"quantity": "adaptive stop threshold", "value": cfg.threshold},
        {"quantity": "probability of stopping early", "value": fs["stop_rate"]},
        {"quantity": "average stopping time", "value": fs["avg_stop_time"]},
        {"quantity": "average fixed-time mean", "value": fs["fixed_mean_avg"]},
        {"quantity": "average stopped mean", "value": fs["stopped_mean_avg"]},
        {"quantity": "fixed-time Hoeffding coverage", "value": fs["fixed_cover"]},
        {"quantity": "stopped-time naive Hoeffding coverage", "value": fs["stopped_cover_naive"]},
        {"quantity": "stopped-time union-bound Hoeffding coverage", "value": fs["stopped_cover_union"]},
        {"quantity": "one final look false alarm with normal bar", "value": fs["final_normal_false_alarm"]},
        {"quantity": "many peeks false alarm with same normal bar", "value": fs["anytime_normal_false_alarm"]},
        {"quantity": "many peeks false alarm with union bar", "value": fs["anytime_union_false_alarm"]},
        {"quantity": "post-selection average of one fixed arm", "value": ps["ordinary_mean_avg"]},
        {"quantity": "post-selection average of empirical winner", "value": ps["selected_mean_avg"]},
        {"quantity": "post-selection bias", "value": ps["selection_bias"]},
    ]
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "results.csv"), index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
