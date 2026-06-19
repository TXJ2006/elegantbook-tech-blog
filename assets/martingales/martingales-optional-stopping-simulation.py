"""Reproducible experiments for
'Martingales and Optional Stopping: Learning at a Random Time'.

The script creates:
  - stopped_random_walks.pdf/png
  - peeking_false_alarms.pdf/png
  - eprocess_paths.pdf/png
  - jackpot_martingale.pdf/png
  - martingale_results.csv

All simulations use a fixed random seed.
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
SEED = 20260618


@dataclass(frozen=True)
class Config:
    alpha: float = 0.05
    p0: float = 0.50
    p1: float = 0.60
    horizon: int = 300
    runs: int = 50000
    random_walk_runs: int = 100000
    random_walk_horizon: int = 500
    random_walk_boundary: int = 12


def first_crossing_stopped_random_walk(
    rng: np.random.Generator,
    runs: int,
    horizon: int,
    boundary: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate symmetric random walks stopped at +/- boundary or horizon."""
    positions = np.zeros(runs, dtype=np.int32)
    stopped = np.zeros(runs, dtype=bool)
    tau = np.full(runs, horizon, dtype=np.int32)

    for t in range(1, horizon + 1):
        active = ~stopped
        if not np.any(active):
            break
        steps = np.where(rng.random(np.sum(active)) < 0.5, -1, 1)
        positions[active] += steps
        crossed = active & (np.abs(positions) >= boundary)
        tau[crossed] = t
        stopped[crossed] = True

    return positions, tau, stopped


def sample_stopped_paths(
    rng: np.random.Generator,
    n_paths: int,
    horizon: int,
    boundary: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    paths: list[tuple[np.ndarray, np.ndarray]] = []
    for _ in range(n_paths):
        s = 0
        values = [0]
        times = [0]
        for t in range(1, horizon + 1):
            s += 1 if rng.random() < 0.5 else -1
            values.append(s)
            times.append(t)
            if abs(s) >= boundary:
                break
        paths.append((np.asarray(times), np.asarray(values)))
    return paths


def exact_one_sided_thresholds(horizon: int, alpha: float, p0: float) -> np.ndarray:
    """k_t such that P_{p0}(Bin(t,p0) >= k_t) <= alpha."""
    thresholds = np.empty(horizon + 1, dtype=np.int32)
    thresholds[0] = 1
    for t in range(1, horizon + 1):
        # scipy's isf returns x with P(X > x) <= alpha, so k = x + 1.
        thresholds[t] = int(binom.isf(alpha, t, p0)) + 1
    return thresholds


def sequential_testing_rates(
    rng: np.random.Generator,
    p: float,
    cfg: Config,
    thresholds: np.ndarray,
) -> dict[str, float]:
    """Compare fixed-horizon, repeated fixed-time, and e-process tests."""
    n = cfg.runs
    successes = np.zeros(n, dtype=np.int32)
    repeated_alarm = np.zeros(n, dtype=bool)
    e_alarm = np.zeros(n, dtype=bool)
    log_e = np.zeros(n, dtype=float)
    log_threshold = math.log(1.0 / cfg.alpha)
    log_success = math.log(cfg.p1 / cfg.p0)
    log_failure = math.log((1.0 - cfg.p1) / (1.0 - cfg.p0))

    for t in range(1, cfg.horizon + 1):
        x = rng.random(n) < p
        successes += x
        repeated_alarm |= successes >= thresholds[t]
        log_e += np.where(x, log_success, log_failure)
        e_alarm |= log_e >= log_threshold

    fixed_alarm = successes >= thresholds[cfg.horizon]
    return {
        "fixed_horizon": float(np.mean(fixed_alarm)),
        "repeated_fixed_time": float(np.mean(repeated_alarm)),
        "e_process": float(np.mean(e_alarm)),
    }


def make_eprocess_paths(
    rng: np.random.Generator,
    p: float,
    p0: float,
    p1: float,
    horizon: int,
    n_paths: int,
) -> np.ndarray:
    x = rng.random((n_paths, horizon)) < p
    increments = np.where(x, math.log(p1 / p0), math.log((1 - p1) / (1 - p0)))
    return np.exp(np.cumsum(increments, axis=1))


def jackpot_paths(rng: np.random.Generator, n_paths: int, horizon: int) -> np.ndarray:
    """M_t = 2^t 1{the first t tosses are all heads}."""
    values = np.ones((n_paths, horizon + 1), dtype=float)
    alive = np.ones(n_paths, dtype=bool)
    for t in range(1, horizon + 1):
        heads = rng.random(n_paths) < 0.5
        alive &= heads
        values[:, t] = np.where(alive, 2.0**t, 0.0)
    return values


def plot_stopped_random_walks(paths: list[tuple[np.ndarray, np.ndarray]], boundary: int) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    for times, values in paths:
        ax.plot(times, values, linewidth=1.0, alpha=0.8)
        ax.scatter(times[-1], values[-1], s=15)
    ax.axhline(boundary, linestyle="--", linewidth=1.2)
    ax.axhline(-boundary, linestyle="--", linewidth=1.2)
    ax.axhline(0, linewidth=0.8)
    ax.set_xlabel("time")
    ax.set_ylabel("martingale value $S_t$")
    ax.set_title("A fair random walk stopped when it reaches a boundary")
    ax.text(0.985, 0.955, fr"boundaries $\pm {boundary}$", transform=ax.transAxes, ha="right", va="top", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "stopped_random_walks.pdf", bbox_inches="tight")
    fig.savefig(OUT / "stopped_random_walks.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_testing_rates(null_rates: dict[str, float], alt_rates: dict[str, float]) -> None:
    labels = ["fixed horizon", "peek repeatedly", "e-process"]
    keys = ["fixed_horizon", "repeated_fixed_time", "e_process"]
    x = np.arange(len(labels))
    width = 0.36

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.bar(x - width / 2, [null_rates[k] for k in keys], width, label="null: $p=0.5$")
    ax.bar(x + width / 2, [alt_rates[k] for k in keys], width, label="alternative: $p=0.6$")
    ax.axhline(0.05, linestyle="--", linewidth=1.1, label="nominal level $0.05$")
    ax.set_xticks(x, labels)
    ax.set_ylabel("probability of stopping and rejecting")
    ax.set_ylim(0, 1.12)
    ax.set_title("Repeated use of a fixed-time test creates false alarms", pad=34)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.105), fontsize=8)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", padding=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "peeking_false_alarms.pdf", bbox_inches="tight")
    fig.savefig(OUT / "peeking_false_alarms.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_eprocess_paths(null_paths: np.ndarray, alt_paths: np.ndarray, alpha: float) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    t = np.arange(1, null_paths.shape[1] + 1)
    for i, path in enumerate(null_paths):
        ax.plot(t, path, linewidth=0.9, alpha=0.55, color="0.55", label="null $p=0.5$" if i == 0 else None)
    for i, path in enumerate(alt_paths):
        ax.plot(t, path, linewidth=1.1, alpha=0.75, color="C0", label="alternative $p=0.6$" if i == 0 else None)
    ax.axhline(1 / alpha, linestyle="--", linewidth=1.2, color="C1", label="$1/\\alpha$")
    ax.set_yscale("log")
    ax.set_xlabel("time")
    ax.set_ylabel("e-process $L_t$ (log scale)")
    ax.set_title("Evidence may be monitored continuously without changing the threshold")
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "eprocess_paths.pdf", bbox_inches="tight")
    fig.savefig(OUT / "eprocess_paths.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_jackpot_martingale(paths: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.25))
    t = np.arange(paths.shape[1])
    for path in paths:
        ax.step(t, path, where="post", linewidth=1.2, alpha=0.8)
    ax.set_yscale("symlog", linthresh=0.5)
    ax.set_xlabel("time")
    ax.set_ylabel(r"$M_t = 2^t \,\mathbf{1}\{\mathrm{all\ heads\ so\ far}\}$")
    ax.set_title("A martingale whose infinite stopping limit loses its mean")
    ax.text(0.98, 0.92, "every finite-time mean is 1\nfirst tail sends the path to 0", 
            transform=ax.transAxes, ha="right", va="top", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "jackpot_martingale.pdf", bbox_inches="tight")
    fig.savefig(OUT / "jackpot_martingale.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    cfg = Config()
    rng = np.random.default_rng(SEED)

    positions, tau, hit = first_crossing_stopped_random_walk(
        rng,
        cfg.random_walk_runs,
        cfg.random_walk_horizon,
        cfg.random_walk_boundary,
    )
    paths = sample_stopped_paths(
        rng,
        n_paths=14,
        horizon=120,
        boundary=cfg.random_walk_boundary,
    )
    plot_stopped_random_walks(paths, cfg.random_walk_boundary)

    thresholds = exact_one_sided_thresholds(cfg.horizon, cfg.alpha, cfg.p0)
    null_rates = sequential_testing_rates(rng, cfg.p0, cfg, thresholds)
    alt_rates = sequential_testing_rates(rng, cfg.p1, cfg, thresholds)
    plot_testing_rates(null_rates, alt_rates)

    null_paths = make_eprocess_paths(rng, cfg.p0, cfg.p0, cfg.p1, 250, 8)
    alt_paths = make_eprocess_paths(rng, cfg.p1, cfg.p0, cfg.p1, 250, 8)
    plot_eprocess_paths(null_paths, alt_paths, cfg.alpha)

    jackpot = jackpot_paths(rng, n_paths=10, horizon=12)
    plot_jackpot_martingale(jackpot)

    upper_hits = np.mean(positions == cfg.random_walk_boundary)
    lower_hits = np.mean(positions == -cfg.random_walk_boundary)
    results = pd.DataFrame(
        [
            {"experiment": "bounded stopping", "metric": "mean stopped value", "value": float(np.mean(positions))},
            {"experiment": "bounded stopping", "metric": "upper-boundary probability", "value": float(upper_hits)},
            {"experiment": "bounded stopping", "metric": "lower-boundary probability", "value": float(lower_hits)},
            {"experiment": "bounded stopping", "metric": "mean stopping time", "value": float(np.mean(tau))},
            {"experiment": "null p=0.5", "metric": "fixed-horizon rejection", "value": null_rates["fixed_horizon"]},
            {"experiment": "null p=0.5", "metric": "repeated fixed-time rejection", "value": null_rates["repeated_fixed_time"]},
            {"experiment": "null p=0.5", "metric": "e-process rejection", "value": null_rates["e_process"]},
            {"experiment": "alternative p=0.6", "metric": "fixed-horizon rejection", "value": alt_rates["fixed_horizon"]},
            {"experiment": "alternative p=0.6", "metric": "repeated fixed-time rejection", "value": alt_rates["repeated_fixed_time"]},
            {"experiment": "alternative p=0.6", "metric": "e-process rejection", "value": alt_rates["e_process"]},
        ]
    )
    results.to_csv(OUT / "martingale_results.csv", index=False)
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
