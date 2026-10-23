"""Sparsity sweep: Taylor-saliency vs magnitude pruning, across seeds, with real cost.

For every seed and target sparsity, trains the full prune_mlp schedule once
per criterion. The two criteria share everything else (data, split, init,
batch order, schedule, safeguards), so each seed gives a *paired* comparison.

Accuracy is measured on the 300-sample test split and on a fresh 100k sample
from the same distribution, whose evaluation noise is negligible. Cost is
measured on forward paths that skip pruned weights (``nn/sparse.py``).

    python -m pytrainer.experiments.sweep --out-dir results/sweep
"""

import os

# Single-threaded BLAS: one run per core, and fair wall-clock timing.
for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(var, "1")

import argparse
import csv
import timeit
from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn.datasets import make_moons

from pytrainer.experiments.data import prepare_two_moons
from pytrainer.experiments.metrics import evaluate
from pytrainer.experiments.prune_mlp import PruneConfig, train_with_pruning
from pytrainer.nn.sparse import SparseLayer, compact, dense, forward, macs, masked_layers

SPARSITIES = (0.0, 0.5, 0.75, 0.9, 0.95)
CRITERIA = ("taylor_abs", "magnitude")
# The default 5% per-layer floor keeps 32 weights, more than the 30 that 95%
# allows, so the whole sweep uses a 2% floor (same for both criteria).
MIN_KEEP_FRACTION = 0.02
FRESH_N = 100_000
FRESH_SEED = 1_000_000  # outside the training seed range
BATCH_SIZES = (1, 256, FRESH_N)
COLORS = {"taylor_abs": "#2a78d6", "magnitude": "#eb6834"}
LABELS = {"taylor_abs": "Taylor |w·g|", "magnitude": "magnitude |w|"}
INK, INK_2, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"


def fresh_sample():
    return make_moons(n_samples=FRESH_N, noise=0.2, random_state=FRESH_SEED)


def run_one(job):
    """Train one (seed, criterion, sparsity); return its metrics and pruned layers."""
    seed, criterion, sparsity = job
    data = prepare_two_moons(seed=seed)
    config = PruneConfig(seed=seed, criterion=criterion, final_sparsity=sparsity,
                         min_keep_fraction=MIN_KEEP_FRACTION, log_every=0)
    res = train_with_pruning(config, data)
    model, pruner = res["model"], res["pruner"]

    X_fresh, y_fresh = fresh_sample()
    X_fresh = (X_fresh - data.mean) / data.std
    test = evaluate(model, data.X_test, data.y_test)
    fresh = evaluate(model, X_fresh, y_fresh)

    # The cheap paths must compute the same function as the trained model.
    layers = masked_layers(model)
    sparse = [SparseLayer(w, b) for w, b in layers]
    compacted = compact(layers)
    reference = model(data.X_val).data
    path_error = max(np.abs(forward(p, data.X_val) - reference).max()
                     for p in (sparse, [dense(w, b) for w, b in compacted]))
    if path_error > 1e-9:
        raise AssertionError(f"sparse paths diverge from the model by {path_error}")

    kept = sum(r["kept"] for r in pruner.layer_report().values())
    row = {
        "seed": seed,
        "criterion": criterion if sparsity else "dense",
        "target_sparsity": sparsity,
        "achieved_sparsity": pruner.sparsity(),
        "kept_weights": kept,
        "selected_epoch": res["best_epoch"],
        "val_accuracy": res["history"]["val_accuracy"][res["best_epoch"]],
        "test_accuracy": test["accuracy"],
        "test_loss": test["loss"],
        "fresh_accuracy": fresh["accuracy"],
        "fresh_loss": fresh["loss"],
        "safeguard_kept_max": res["safeguard_kept_max"],
        "dense_macs": macs(layers),
        "sparse_macs": sum(layer.nnz for layer in sparse),
        "compact_macs": macs(compacted),
        "compact_shape": "-".join(str(w.shape[0]) for w, _ in compacted) + "-2",
        "max_path_error": path_error,
    }
    return row, layers


def write_csv(rows, path):
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def mean_ci(values, n_boot=10_000):
    """Mean and 95% percentile-bootstrap interval over seeds.

    Bootstrap rather than a t-interval: a few seeds collapse at high
    sparsity, so the per-seed values are far from normal.
    """
    values = np.asarray(values, dtype=float)
    resampled = np.random.default_rng(0).choice(values, (n_boot, values.size)).mean(axis=1)
    low, high = np.percentile(resampled, [2.5, 97.5])
    return values.mean(), low, high


def group(rows, criterion, sparsity):
    """Rows for one curve point, sorted by seed; the dense runs belong to both curves."""
    name = "dense" if sparsity == 0 else criterion
    return sorted((r for r in rows if r["criterion"] == name and r["target_sparsity"] == sparsity),
                  key=lambda r: r["seed"])


def summarize(rows):
    summary = []
    for sparsity in SPARSITIES:
        for criterion in (("dense",) if sparsity == 0 else CRITERIA):
            g = group(rows, criterion, sparsity)
            fresh, fresh_low, fresh_high = mean_ci([r["fresh_accuracy"] for r in g])
            test, test_low, test_high = mean_ci([r["test_accuracy"] for r in g])
            summary.append({
                "criterion": criterion, "target_sparsity": sparsity, "n_seeds": len(g),
                "achieved_sparsity": np.mean([r["achieved_sparsity"] for r in g]),
                "kept_weights": np.mean([r["kept_weights"] for r in g]),
                "fresh_acc_mean": fresh, "fresh_acc_ci95_low": fresh_low,
                "fresh_acc_ci95_high": fresh_high,
                "fresh_acc_sd": np.std([r["fresh_accuracy"] for r in g], ddof=1),
                "fresh_acc_min": min(r["fresh_accuracy"] for r in g),
                "test_acc_mean": test, "test_acc_ci95_low": test_low,
                "test_acc_ci95_high": test_high,
                "sparse_macs": np.mean([r["sparse_macs"] for r in g]),
                "compact_macs": np.mean([r["compact_macs"] for r in g]),
                "safeguard_kept_max": max(r["safeguard_kept_max"] for r in g),
            })
    return summary


def compare(rows):
    """Paired Taylor - magnitude differences per sparsity (same seed = same data and init)."""
    out = []
    for sparsity in SPARSITIES[1:]:
        taylor, magnitude = (group(rows, c, sparsity) for c in CRITERIA)
        assert [r["seed"] for r in taylor] == [r["seed"] for r in magnitude]
        for metric in ("fresh_accuracy", "test_accuracy", "fresh_loss"):
            a = np.array([r[metric] for r in taylor])
            b = np.array([r[metric] for r in magnitude])
            d = a - b
            mean, low, high = mean_ci(d)
            nonzero = d[d != 0]
            out.append({
                "target_sparsity": sparsity, "metric": metric, "n_seeds": len(d),
                "taylor_mean": a.mean(), "magnitude_mean": b.mean(),
                "diff_mean": mean, "diff_ci95_low": low, "diff_ci95_high": high,
                "diff_sd": d.std(ddof=1),
                "taylor_wins": int(np.sum(d > 0)), "ties": int(np.sum(d == 0)),
                "magnitude_wins": int(np.sum(d < 0)),
                "paired_t_p": stats.ttest_rel(a, b).pvalue,
                "wilcoxon_p": stats.wilcoxon(nonzero).pvalue if nonzero.size else 1.0,
            })
    return out


def benchmark(layers_by_sparsity, X):
    """Wall-clock µs/sample for dense-with-zeros vs sparse vs compacted, single thread."""
    out = []
    for sparsity, layers in layers_by_sparsity.items():
        paths = {
            "dense_with_zeros": [dense(w, b) for w, b in layers],
            "sparse": [SparseLayer(w, b) for w, b in layers],
            "compact_dense": [dense(w, b) for w, b in compact(layers)],
        }
        for batch in BATCH_SIZES:
            x = X[:batch]
            times = {}
            for name, path in paths.items():
                timer = timeit.Timer(lambda: forward(path, x))
                number, _ = timer.autorange()
                times[name] = min(timer.repeat(5, number)) / number / batch * 1e6
            for name, t in times.items():
                out.append({"target_sparsity": sparsity, "batch_size": batch, "path": name,
                            "us_per_sample": t,
                            "speedup_vs_dense_with_zeros": times["dense_with_zeros"] / t})
    return out


def plot(rows, comparison, path):
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 4.8), facecolor=SURFACE,
                                  gridspec_kw={"width_ratios": [3, 2]})
    rng = np.random.default_rng(0)
    for a in (ax, ax2):
        a.set_facecolor(SURFACE)
        a.grid(color="#e4e3df", linewidth=0.8)
        a.tick_params(colors=INK_2)
        for spine in a.spines.values():
            spine.set_color("#c9c8c2")
        a.spines[["top", "right"]].set_visible(False)

    for k, criterion in enumerate(CRITERIA):
        xs, means, errs = [], [], [[], []]
        for sparsity in SPARSITIES:
            g = group(rows, criterion, sparsity)
            acc = 100 * np.array([r["fresh_accuracy"] for r in g])
            x = 100 * np.mean([r["achieved_sparsity"] for r in g])
            offset = (k - 0.5) * 0.8 if sparsity else 0.0  # separate the two curves' seed dots
            ax.scatter(x + offset + rng.uniform(-0.3, 0.3, acc.size), acc, s=14,
                       color=COLORS[criterion], alpha=0.45, linewidths=0)
            m, low, high = mean_ci(acc)
            xs.append(x + offset); means.append(m)
            errs[0].append(m - low); errs[1].append(high - m)
        ax.errorbar(xs, means, yerr=errs, color=COLORS[criterion], linewidth=2,
                    marker="o" if k == 0 else "s", markersize=6, capsize=3,
                    label=f"{LABELS[criterion]} (mean, 95% bootstrap CI)")
        ax.annotate(LABELS[criterion], (xs[-1], means[-1]), xytext=(8, -4 + 10 * (k == 0)),
                    textcoords="offset points", color=INK, fontsize=9)
    ax.set_xlabel("weights removed (% of 608 eligible; biases dense)", color=INK_2)
    ax.set_ylabel("accuracy on fresh 100k sample (%)", color=INK_2)
    n = len(group(rows, "dense", 0.0))
    ax.set_title(f"Accuracy vs sparsity, {n} seeds (dots = individual seeds)",
                 color=INK, loc="left", fontsize=11)
    ax.set_xlim(-3, 104)
    ax.legend(loc="lower left", frameon=False, labelcolor=INK)

    fresh = [c for c in comparison if c["metric"] == "fresh_accuracy"]
    for i, c in enumerate(fresh):
        taylor, magnitude = (group(rows, crit, c["target_sparsity"]) for crit in CRITERIA)
        d = 100 * np.array([a["fresh_accuracy"] - b["fresh_accuracy"]
                            for a, b in zip(taylor, magnitude)])
        ax2.scatter(i + rng.uniform(-0.12, 0.12, d.size), d, s=12, color=INK_2,
                    alpha=0.45, linewidths=0)
        yerr = [[100 * (c["diff_mean"] - c["diff_ci95_low"])], [100 * (c["diff_ci95_high"] - c["diff_mean"])]]
        ax2.errorbar(i, 100 * c["diff_mean"], yerr=yerr,
                     color=INK, marker="D", markersize=6, capsize=4, linewidth=2)
        # Small effects are invisible at the 95% scale, so print every one.
        ax2.annotate(f"{100 * c['diff_mean']:+.2f} pp\n[{100 * c['diff_ci95_low']:+.2f}, "
                     f"{100 * c['diff_ci95_high']:+.2f}]",
                     (i, 100 * c["diff_ci95_high"]), xytext=(0, 8), textcoords="offset points",
                     ha="center", va="bottom", color=INK, fontsize=8)
    ax2.axhline(0, color=INK_2, linewidth=1)
    ax2.set_xticks(range(len(fresh)), [f"{100 * c['target_sparsity']:.0f}%" for c in fresh])
    ax2.set_xlabel("target sparsity", color=INK_2)
    ax2.set_ylabel("Taylor − magnitude accuracy (pp)", color=INK_2)
    ax2.set_title("Paired difference per seed (> 0: Taylor better)",
                  color=INK, loc="left", fontsize=11)

    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser(description="Sparsity sweep, Taylor vs magnitude, across seeds")
    p.add_argument("--out-dir", default="results/sweep")
    p.add_argument("--seeds", type=int, default=30)
    p.add_argument("--first-seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=os.cpu_count())
    a = p.parse_args(argv)
    os.makedirs(a.out_dir, exist_ok=True)

    seeds = range(a.first_seed, a.first_seed + a.seeds)
    jobs = [(seed, "taylor_abs", 0.0) for seed in seeds]  # dense: criterion unused
    jobs += [(seed, c, s) for s in SPARSITIES[1:] for c in CRITERIA for seed in seeds]
    print(f"{len(jobs)} training runs on {a.workers} workers...")
    with Pool(a.workers) as pool:
        results = pool.map(run_one, jobs, chunksize=1)
    rows = [row for row, _ in results]

    write_csv(rows, os.path.join(a.out_dir, "runs.csv"))
    summary = summarize(rows)
    write_csv(summary, os.path.join(a.out_dir, "summary.csv"))
    comparison = compare(rows)
    write_csv(comparison, os.path.join(a.out_dir, "comparison.csv"))
    plot(rows, comparison, os.path.join(a.out_dir, "pareto.png"))

    # Cost: first seed's Taylor model at every sparsity (dense model at 0%).
    layers_by_sparsity = {row["target_sparsity"]: layers for row, layers in results
                          if row["seed"] == a.first_seed and row["criterion"] in ("dense", "taylor_abs")}
    X_fresh, _ = fresh_sample()
    write_csv(benchmark(layers_by_sparsity, X_fresh), os.path.join(a.out_dir, "cost.csv"))
    print(f"wrote runs.csv, summary.csv, comparison.csv, cost.csv, pareto.png to {a.out_dir}")


if __name__ == "__main__":
    main()
