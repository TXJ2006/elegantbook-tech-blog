
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def run_once(policy, probs, horizon, rng, epsilon=0.10):
    K = len(probs)
    counts = np.zeros(K, dtype=int)
    sums = np.zeros(K, dtype=float)
    alpha = np.ones(K)
    beta = np.ones(K)
    mu_star = np.max(probs)
    cumulative_regret = np.zeros(horizon)

    def pull(a):
        r = rng.binomial(1, probs[a])
        counts[a] += 1
        sums[a] += r
        alpha[a] += r
        beta[a] += 1 - r
        return r

    regret = 0.0

    # give each policy one clean round of initial data, except Thompson, which can act from its prior.
    if policy in ["Greedy", "Epsilon-greedy", "UCB"]:
        for a in range(K):
            if a >= horizon:
                break
            r = pull(a)
            regret += mu_star - probs[a]
            cumulative_regret[a] = regret
        start = min(K, horizon)
    else:
        start = 0

    for t in range(start, horizon):
        if policy == "Greedy":
            means = np.divide(sums, np.maximum(counts, 1))
            a = int(np.argmax(means))
        elif policy == "Epsilon-greedy":
            if rng.random() < epsilon:
                a = int(rng.integers(K))
            else:
                means = np.divide(sums, np.maximum(counts, 1))
                a = int(np.argmax(means))
        elif policy == "UCB":
            means = sums / counts
            bonus = np.sqrt(2.0 * np.log(max(t + 1, 2)) / counts)
            a = int(np.argmax(means + bonus))
        elif policy == "Thompson sampling":
            theta = rng.beta(alpha, beta)
            a = int(np.argmax(theta))
        else:
            raise ValueError(policy)

        r = pull(a)
        regret += mu_star - probs[a]
        cumulative_regret[t] = regret

    return cumulative_regret, counts

def main():
    probs = np.array([0.30, 0.35, 0.42])
    horizon = 2000
    n_runs = 500
    policies = ["Greedy", "Epsilon-greedy", "UCB", "Thompson sampling"]

    all_regrets = {}
    all_counts = {}
    for policy in policies:
        regrets = []
        counts = []
        for seed in range(n_runs):
            rng = np.random.default_rng(202706 + 1009 * seed + len(policy))
            r, c = run_once(policy, probs, horizon, rng)
            regrets.append(r)
            counts.append(c)
        all_regrets[policy] = np.vstack(regrets)
        all_counts[policy] = np.vstack(counts)

    rows = []
    for policy in policies:
        final = all_regrets[policy][:, -1]
        cnt = all_counts[policy]
        rows.append({
            "policy": policy,
            "mean_final_regret": final.mean(),
            "std_final_regret": final.std(ddof=1),
            "arm0_mean_pulls": cnt[:, 0].mean(),
            "arm1_mean_pulls": cnt[:, 1].mean(),
            "arm2_mean_pulls": cnt[:, 2].mean(),
        })
    df = pd.DataFrame(rows)
    df.to_csv("results.csv", index=False)

    plt.figure(figsize=(7.2, 4.2))
    x = np.arange(1, horizon + 1)
    for policy in policies:
        mean = all_regrets[policy].mean(axis=0)
        se = all_regrets[policy].std(axis=0, ddof=1) / np.sqrt(n_runs)
        plt.plot(x, mean, label=policy)
        plt.fill_between(x, mean - 2 * se, mean + 2 * se, alpha=0.12)
    plt.xlabel("round")
    plt.ylabel("mean cumulative regret")
    plt.title("Regret in a three-arm Bernoulli bandit")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig("regret_curves.pdf")
    plt.savefig("regret_curves.png", dpi=200)
    plt.close()

    plt.figure(figsize=(7.2, 4.2))
    labels = [r"$p=0.30$", r"$p=0.35$", r"$p=0.42$"]
    xloc = np.arange(len(labels))
    width = 0.18
    for i, policy in enumerate(policies):
        means = all_counts[policy].mean(axis=0)
        plt.bar(xloc + (i - 1.5) * width, means, width, label=policy)
    plt.xticks(xloc, labels)
    plt.ylabel("mean number of pulls")
    plt.title("Where each algorithm spends its samples")
    plt.legend(frameon=False, fontsize=8)
    plt.tight_layout()
    plt.savefig("pull_counts.pdf")
    plt.savefig("pull_counts.png", dpi=200)
    plt.close()

if __name__ == "__main__":
    main()
