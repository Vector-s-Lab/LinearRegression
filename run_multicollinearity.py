"""
Controlled Multicollinearity Study (Synthetic Data)

Generates synthetic linear regression data with a known ground truth
and controlled inter-feature correlation structure. Varies the correlation
level to study how each XAI method's ranking accuracy degrades.

Ground truth: y = w1*x1 + w2*x2 + ... + wn*xn + noise
True importance ranking is known from the weights.

Usage:
    python run_multicollinearity.py
    python run_multicollinearity.py --n-features 8 --n-samples 1000
    python run_multicollinearity.py --levels 0.0 0.3 0.5 0.7 0.9 0.95
    python run_multicollinearity.py --kpi r2
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

from benchmark import (
    build_char_func,
    build_shapg_custom_function,
    evaluate_feature_dropping,
    run_explainers,
    save_results,
)
from benchmark.plotting import plot_feature_dropping, DISPLAY_NAMES

import matplotlib.pyplot as plt

try:
    import gnuplot_style as gp
    gp.use("all")
except ImportError:
    pass


def build_block_covariance(n_features, block_sizes, correlation_level):
    """Build a block-correlation covariance matrix.

    Features within the same block have pairwise correlation = correlation_level.
    Features in different blocks are independent (correlation = 0).

    Parameters
    ----------
    n_features : int
    block_sizes : list of int
        Size of each block, must sum to n_features.
        E.g. [4, 4] for two blocks of 4 features.
    correlation_level : float

    Returns
    -------
    cov : ndarray of shape (n_features, n_features)
    """
    assert sum(block_sizes) == n_features
    cov = np.eye(n_features)
    offset = 0
    for size in block_sizes:
        block = np.full((size, size), correlation_level)
        np.fill_diagonal(block, 1.0)
        cov[offset:offset + size, offset:offset + size] = block
        offset += size
    return cov


def generate_synthetic_data(n_samples, n_features, correlation_level, weights,
                            block_sizes=None, snr=1.5, random_state=42):
    """Generate synthetic regression data with controlled multicollinearity.

    Features are drawn from a multivariate normal with block-correlation
    structure: features within the same block share pairwise correlation
    = correlation_level, while features across blocks are independent.

    Noise is scaled to maintain a fixed signal-to-noise ratio (SNR):
        noise_std = std(X @ w) / snr

    With SNR=1.5, R² ≈ 0.69 (comparable to real datasets: abalone 0.53,
    concrete 0.63, admission 0.82).

    Parameters
    ----------
    n_samples : int
    n_features : int
    correlation_level : float
        Pairwise correlation within each block (0 = independent, 1 = identical).
    weights : array-like
        True coefficients for y = X @ w + noise. Defines ground truth importance.
    block_sizes : list of int or None
        Sizes of correlated blocks. If None, defaults to two equal blocks.
    snr : float
        Signal-to-noise ratio. Higher = less noise relative to signal.
    random_state : int

    Returns
    -------
    X : DataFrame with columns x1, x2, ..., xn
    y : Series
    actual_noise_std : float
    """
    rng = np.random.default_rng(random_state)

    if block_sizes is None:
        half = n_features // 2
        block_sizes = [half, n_features - half]

    cov = build_block_covariance(n_features, block_sizes, correlation_level)

    X_arr = rng.multivariate_normal(
        mean=np.zeros(n_features), cov=cov, size=n_samples
    )

    # Compute signal and scale noise to maintain constant SNR
    signal = X_arr @ np.array(weights)
    noise_std = np.std(signal) / snr
    noise = rng.normal(0, noise_std, size=n_samples)
    y_arr = signal + noise

    feature_names = [f"x{i+1}" for i in range(n_features)]
    X = pd.DataFrame(X_arr, columns=feature_names)
    y = pd.Series(y_arr, name="y")

    return X, y, noise_std


def get_ground_truth_ranking(weights):
    """Return feature names sorted by |weight| descending (true importance)."""
    indexed = [(f"x{i+1}", abs(w)) for i, w in enumerate(weights)]
    indexed.sort(key=lambda x: x[1], reverse=True)
    return [name for name, _ in indexed]


def compute_kendall_tau(ranking_a, ranking_b):
    """Kendall tau between two ordered lists of feature names."""
    common = [f for f in ranking_a if f in ranking_b]
    if len(common) < 2:
        return 0.0
    rank_a = [ranking_a.index(f) for f in common]
    rank_b = [ranking_b.index(f) for f in common]
    tau, _ = kendalltau(rank_a, rank_b)
    return tau


def make_cfg(kpi, seeds, n_features):
    """Build a config dict for synthetic data experiments."""
    return {
        "dataset": "synthetic",
        "model": {
            "type": "LinearRegression",
            "params": {},
            "train_test_split": {"test_size": 0.2, "random_state": 42},
        },
        "characteristic_function": kpi,
        "explainers": {
            "exact_cis": {"enabled": True},
            "exact_shapley": {"enabled": True},
            "shap_explainer": {"enabled": True},
            "sampling_shap": {"enabled": True},
            "kernel_shap": {"enabled": True},
            "original_shapg": {"enabled": True, "params": {"corr_method": "cosine", "depth": 1, "n_samples": 3}},
            "improved_shapg": {"enabled": True, "params": {"corr_method": "cosine", "sim_method": "cosine", "density_ratio": None}},
        },
        "evaluation": {"drop_limit": min(n_features - 1, 7), "weighted_slope_beta": 0.8},
        "seed": seeds[0],
        "n_runs": len(seeds),
        "plot": {"figsize": [10, 6], "dpi": 300, "format": ["pdf"]},
    }


def main():
    parser = argparse.ArgumentParser(description="Controlled multicollinearity study (synthetic data)")
    parser.add_argument("--n-features", type=int, default=8)
    parser.add_argument("--n-samples", type=int, default=1000)
    parser.add_argument("--snr", type=float, default=2.38, help="Signal-to-noise ratio (2.38 gives R2~0.85)")
    parser.add_argument("--blocks", nargs="+", type=int, default=None,
                        help="Block sizes for correlation structure (default: two equal blocks)")
    parser.add_argument("--kpi", default="r2", choices=["r2", "adj_r2", "aic", "f_statistic"])
    parser.add_argument("--levels", nargs="+", type=float, default=[0.0, 0.3, 0.5, 0.7, 0.9, 0.95])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-runs", type=int, default=10)
    parser.add_argument("--output", type=str, default=None, help="Override output directory")
    args = parser.parse_args()
    args.seeds = [args.seed + i for i in range(args.n_runs)]

    if args.output:
        output_base = Path(args.output)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_base = Path("results") / f"multicollinearity_{args.kpi}_{timestamp}"
    output_base.mkdir(parents=True, exist_ok=True)

    # Ground truth weights — deliberately varied so true ranking is unambiguous
    # w = [8, 6, 4, 3, 2, 1, 0.5, 0.1] for 8 features
    weights = np.linspace(args.n_features, 0.1, args.n_features).tolist()
    ground_truth = get_ground_truth_ranking(weights)

    cfg = make_cfg(args.kpi, args.seeds, args.n_features)

    block_sizes = args.blocks
    if block_sizes is None:
        half = args.n_features // 2
        block_sizes = [half, args.n_features - half]

    print(f"Synthetic data: {args.n_samples} samples, {args.n_features} features")
    print(f"Block structure: {block_sizes} (features within each block are correlated)")
    print(f"Ground truth weights: {[f'{w:.2f}' for w in weights]}")
    print(f"Ground truth ranking: {ground_truth}")
    print(f"KPI: {args.kpi}")
    print(f"Correlation levels: {args.levels}")
    print(f"SNR: {args.snr} (expected R2 ~ {args.snr**2 / (args.snr**2 + 1):.2f})")

    all_level_results = {}

    for level in args.levels:
        print(f"\n{'='*60}")
        print(f"Correlation level = {level}")
        print(f"{'='*60}")

        X, y, actual_noise_std = generate_synthetic_data(
            args.n_samples, args.n_features, level, weights,
            block_sizes=block_sizes, snr=args.snr, random_state=42
        )

        # Verify actual correlation and noise
        actual_corr = X.corr().values
        mask = ~np.eye(args.n_features, dtype=bool)
        mean_all = actual_corr[mask].mean()
        # Within-block mean
        offset = 0
        within_corrs = []
        for size in block_sizes:
            block = actual_corr[offset:offset+size, offset:offset+size]
            block_mask = ~np.eye(size, dtype=bool)
            within_corrs.extend(block[block_mask].tolist())
            offset += size
        mean_within = np.mean(within_corrs) if within_corrs else 0.0
        print(f"  Within-block correlation: {mean_within:.4f} (target: {level})")
        print(f"  Overall mean correlation: {mean_all:.4f}")
        print(f"  Noise std: {actual_noise_std:.4f} (SNR={args.snr})")

        char_func, X_train, X_test, y_train, y_test = build_char_func(cfg, X, y)
        shapg_custom_func = build_shapg_custom_function(cfg, X, y)

        explainer_results = run_explainers(
            cfg, X, y, char_func, X_train, X_test, y_train, y_test, shapg_custom_func
        )
        kpi_results, initial_kpi = evaluate_feature_dropping(cfg, X, y, explainer_results)

        # Save per-level results
        level_dir = output_base / f"level_{level:.2f}"
        save_results(level_dir, explainer_results, kpi_results, initial_kpi, cfg)
        plot_feature_dropping(cfg, kpi_results, level_dir)

        all_level_results[level] = {
            "rankings": {name: [f for f, _ in r["values"]] for name, r in explainer_results.items()},
            "slopes": {name: kpi["slope"] for name, kpi in kpi_results.items()},
            "slopes_std": {name: kpi.get("slope_std", 0.0) for name, kpi in kpi_results.items()},
            "slopes_per_seed": {name: kpi.get("slope_per_seed", [kpi["slope"]]) for name, kpi in kpi_results.items()},
            "timing": {name: r["mean_duration"] for name, r in explainer_results.items()},
        }

    # --- Ranking accuracy: Kendall tau vs ground truth ---
    methods = list(all_level_results[args.levels[0]]["rankings"].keys())

    accuracy = {method: {} for method in methods}
    for level in args.levels:
        for method in methods:
            tau = compute_kendall_tau(ground_truth, all_level_results[level]["rankings"][method])
            accuracy[method][level] = tau

    print(f"\n{'='*60}")
    print("RANKING ACCURACY (Kendall tau vs ground truth)")
    print(f"{'='*60}")
    print(f"\n{'Method':<25}", end="")
    for level in args.levels:
        print(f"  r={level:<6}", end="")
    print()
    print("-" * (25 + 10 * len(args.levels)))
    for method in methods:
        print(f"{method:<25}", end="")
        for level in args.levels:
            print(f"  {accuracy[method][level]:>6.3f}", end="")
        print()

    # --- Ranking stability: Kendall tau vs rho=0 baseline ---
    baseline_level = args.levels[0]
    baseline_rankings = all_level_results[baseline_level]["rankings"]

    stability = {method: {} for method in methods}
    for level in args.levels[1:]:
        for method in methods:
            tau = compute_kendall_tau(
                baseline_rankings[method],
                all_level_results[level]["rankings"][method]
            )
            stability[method][level] = tau

    print(f"\n{'='*60}")
    print(f"RANKING STABILITY (Kendall tau vs rho={baseline_level} baseline)")
    print(f"{'='*60}")
    print(f"\n{'Method':<25}", end="")
    for level in args.levels[1:]:
        print(f"  r={level:<6}", end="")
    print()
    print("-" * (25 + 10 * len(args.levels[1:])))
    for method in methods:
        print(f"{method:<25}", end="")
        for level in args.levels[1:]:
            print(f"  {stability[method][level]:>6.3f}", end="")
        print()

    # --- Plot 1: Ranking accuracy vs ground truth ---
    fig, ax = plt.subplots(figsize=(10, 6))
    for method in methods:
        taus = [accuracy[method][l] for l in args.levels]
        ax.plot(args.levels, taus, label=DISPLAY_NAMES.get(method, method), linewidth=1.5, alpha=0.7)

    ax.set_xlabel("Pairwise Feature Correlation ($\\rho$)", fontsize=12)
    ax.set_ylabel("Kendall $\\tau$ (vs ground truth)", fontsize=12)
    ax.set_title(f"Ranking Accuracy under Multicollinearity (n={args.n_samples}, p={args.n_features}, SNR={args.snr}, {args.kpi})")
    ax.set_ylim(-0.2, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, frameon=True)
    plt.tight_layout()
    for fmt in ["pdf"]:
        plt.savefig(output_base / f"accuracy.{fmt}", format=fmt, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\nAccuracy plot saved to {output_base}/accuracy.pdf")

    # --- Plot 2: Ranking stability vs baseline ---
    fig, ax = plt.subplots(figsize=(10, 6))
    for method in methods:
        taus = [stability[method][l] for l in args.levels[1:]]
        ax.plot(args.levels[1:], taus, label=DISPLAY_NAMES.get(method, method), linewidth=1.5, alpha=0.7)

    ax.set_xlabel("Pairwise Feature Correlation ($\\rho$)", fontsize=12)
    ax.set_ylabel(f"Kendall $\\tau$ (vs $\\rho$={baseline_level} ranking)", fontsize=12)
    ax.set_title(f"Ranking Stability under Multicollinearity (n={args.n_samples}, p={args.n_features}, SNR={args.snr}, {args.kpi})")
    ax.set_ylim(-0.2, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, frameon=True)
    plt.tight_layout()
    for fmt in ["pdf"]:
        plt.savefig(output_base / f"stability.{fmt}", format=fmt, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Stability plot saved to {output_base}/stability.pdf")

    # --- Plot 3: Weighted slopes across levels ---
    fig, ax = plt.subplots(figsize=(10, 6))
    for method in methods:
        slopes = [all_level_results[l]["slopes"][method] for l in args.levels]
        ax.plot(args.levels, slopes, label=DISPLAY_NAMES.get(method, method), linewidth=1.5, alpha=0.7)

    ax.set_xlabel("Pairwise Feature Correlation ($\\rho$)", fontsize=14)
    ax.set_ylabel("Weighted Slope $S$", fontsize=14)
    ax.tick_params(labelsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=14, frameon=True, ncol=2)
    plt.tight_layout()
    for fmt in ["pdf"]:
        plt.savefig(output_base / f"slopes.{fmt}", format=fmt, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Slope plot saved to {output_base}/slopes.pdf")

    # --- Save summary ---
    summary = {
        "n_samples": args.n_samples,
        "n_features": args.n_features,
        "block_sizes": block_sizes,
        "snr": args.snr,
        "kpi": args.kpi,
        "weights": weights,
        "ground_truth_ranking": ground_truth,
        "levels": args.levels,
        "accuracy": {m: {str(l): v for l, v in a.items()} for m, a in accuracy.items()},
        "stability": {m: {str(l): v for l, v in s.items()} for m, s in stability.items()},
        "slopes": {str(l): all_level_results[l]["slopes"] for l in args.levels},
        "slopes_std": {str(l): all_level_results[l]["slopes_std"] for l in args.levels},
        "slopes_per_seed": {str(l): all_level_results[l]["slopes_per_seed"] for l in args.levels},
    }
    with open(output_base / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nAll results saved to {output_base}/")


if __name__ == "__main__":
    main()
