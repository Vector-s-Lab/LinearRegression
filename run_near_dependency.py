"""
Near-dependency multicollinearity regime.

Complements run_multicollinearity.py (which uses block pairwise correlation).
This regime constructs features where a subset are noisy linear copies of
others, producing VIF inflation that matches the severity of real datasets
(Abalone max VIF ~110, UCS-SCM ~597).

Design (n_features = 8)
    x_1, x_2, x_3 : primary, i.i.d. N(0, 1) with weights (8.0, 6.0, 4.0)
    x_4 = x_1 + sigma_d * Z_4   (near-duplicate of x_1, weight 0)
    x_5 = x_2 + sigma_d * Z_5   (near-duplicate of x_2, weight 0)
    x_6 = x_3 + sigma_d * Z_6   (near-duplicate of x_3, weight 0)
    x_7, x_8 : primary, i.i.d. N(0, 1) with weights (3.0, 1.0)

As sigma_d -> 0, x_{i+3} becomes an identical copy of x_i (VIF -> infinity).
As sigma_d -> infinity, the copies become independent (VIF -> 1).

We tune sigma_d via binary search to hit a target max VIF from
{10, 50, 100, 500} -- matching Concrete, moderate, Abalone, UCS-SCM severity.

Usage:
    python run_near_dependency.py
    python run_near_dependency.py --target-vifs 10 50 100 500
    python run_near_dependency.py --kpi r2 --n-runs 10
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

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


PRIMARY_INDICES = [0, 1, 2, 6, 7]   # x_1, x_2, x_3, x_7, x_8
DUPLICATE_MAP = {3: 0, 4: 1, 5: 2}  # x_4<-x_1, x_5<-x_2, x_6<-x_3
WEIGHTS = np.array([8.0, 6.0, 4.0, 0.0, 0.0, 0.0, 3.0, 1.0])


def generate_near_dependency_data(n_samples, sigma_d, random_state=42):
    """Generate synthetic data with near-duplicate columns.

    Returns (X_df, y_series, actual_noise_std, max_vif).
    """
    rng = np.random.default_rng(random_state)
    n_features = len(WEIGHTS)

    X_arr = np.zeros((n_samples, n_features))
    for i in PRIMARY_INDICES:
        X_arr[:, i] = rng.standard_normal(n_samples)
    for dup_idx, src_idx in DUPLICATE_MAP.items():
        X_arr[:, dup_idx] = X_arr[:, src_idx] + sigma_d * rng.standard_normal(n_samples)

    signal = X_arr @ WEIGHTS
    noise_std = np.std(signal) / 2.38
    y_arr = signal + rng.normal(0.0, noise_std, size=n_samples)

    feature_names = [f"x{i+1}" for i in range(n_features)]
    X = pd.DataFrame(X_arr, columns=feature_names)
    y = pd.Series(y_arr, name="y")

    X_c = add_constant(X).values
    vifs = [variance_inflation_factor(X_c, i + 1) for i in range(n_features)]
    return X, y, noise_std, max(vifs)


def calibrate_sigma_for_vif(target_vif, n_samples=1000, random_state=42,
                             lo=1e-4, hi=10.0, tol=0.01, max_iter=40):
    """Binary search for sigma_d that yields max VIF close to target_vif.

    VIF is monotonically decreasing in sigma_d (more noise -> less collinearity).
    """
    _, _, _, vif_hi = generate_near_dependency_data(n_samples, hi, random_state)
    _, _, _, vif_lo = generate_near_dependency_data(n_samples, lo, random_state)
    if target_vif > vif_lo:
        return lo, vif_lo
    if target_vif < vif_hi:
        return hi, vif_hi
    for _ in range(max_iter):
        mid = np.sqrt(lo * hi)
        _, _, _, vif_mid = generate_near_dependency_data(n_samples, mid, random_state)
        if abs(vif_mid - target_vif) / target_vif < tol:
            return mid, vif_mid
        if vif_mid > target_vif:
            lo = mid
        else:
            hi = mid
    return mid, vif_mid


def get_ground_truth_ranking():
    """Features sorted by |weight| descending. Ties in weight 0 broken by index."""
    indexed = [(f"x{i+1}", abs(w), i) for i, w in enumerate(WEIGHTS)]
    indexed.sort(key=lambda t: (-t[1], t[2]))
    return [name for name, _, _ in indexed]


def compute_kendall_tau(ranking_a, ranking_b):
    common = [f for f in ranking_a if f in ranking_b]
    if len(common) < 2:
        return 0.0
    rank_a = [ranking_a.index(f) for f in common]
    rank_b = [ranking_b.index(f) for f in common]
    tau, _ = kendalltau(rank_a, rank_b)
    return tau


def make_cfg(kpi, seeds, n_features):
    return {
        "dataset": "synthetic_near_dep",
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
    parser = argparse.ArgumentParser(description="Near-dependency multicollinearity study")
    parser.add_argument("--target-vifs", nargs="+", type=float, default=[10.0, 50.0, 100.0, 500.0])
    parser.add_argument("--n-samples", type=int, default=1000)
    parser.add_argument("--kpi", default="r2", choices=["r2", "adj_r2", "aic", "f_statistic"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-runs", type=int, default=10)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--calibrate-only", action="store_true",
                        help="Only run sigma_d calibration for each target VIF, don't run XAI")
    args = parser.parse_args()
    args.seeds = [args.seed + i for i in range(args.n_runs)]

    n_features = len(WEIGHTS)
    ground_truth = get_ground_truth_ranking()

    # --- Sigma calibration ---
    print("Calibrating sigma_d for each target max VIF...")
    print(f"{'target VIF':>10}  {'sigma_d':>10}  {'actual VIF':>10}")
    print("-" * 38)
    calib = []
    for target in args.target_vifs:
        sigma, actual = calibrate_sigma_for_vif(target, n_samples=args.n_samples)
        calib.append((target, sigma, actual))
        print(f"{target:>10.1f}  {sigma:>10.4f}  {actual:>10.2f}")

    if args.calibrate_only:
        return

    # --- Set up output ---
    if args.output:
        output_base = Path(args.output)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_base = Path("results") / f"near_dependency_{args.kpi}_{timestamp}"
    output_base.mkdir(parents=True, exist_ok=True)

    cfg = make_cfg(args.kpi, args.seeds, n_features)

    print(f"\nDesign: n_samples={args.n_samples}, p={n_features}")
    print(f"Weights: {WEIGHTS.tolist()}")
    print(f"Primary features: x1, x2, x3, x7, x8 | Near-duplicates: x4(=x1), x5(=x2), x6(=x3)")
    print(f"Ground truth ranking: {ground_truth}")
    print(f"KPI: {args.kpi}  |  n_runs: {args.n_runs}")

    all_results = {}

    for target_vif, sigma, actual_vif in calib:
        print(f"\n{'='*60}")
        print(f"target max VIF = {target_vif}  (sigma_d = {sigma:.4f}, actual max VIF = {actual_vif:.2f})")
        print(f"{'='*60}")

        X, y, noise_std, max_vif = generate_near_dependency_data(
            args.n_samples, sigma, random_state=42
        )
        print(f"  noise_std: {noise_std:.4f}")

        char_func, X_train, X_test, y_train, y_test = build_char_func(cfg, X, y)
        shapg_custom_func = build_shapg_custom_function(cfg, X, y)

        explainer_results = run_explainers(
            cfg, X, y, char_func, X_train, X_test, y_train, y_test, shapg_custom_func
        )
        kpi_results, initial_kpi = evaluate_feature_dropping(cfg, X, y, explainer_results)

        level_dir = output_base / f"vif_{int(target_vif):04d}"
        save_results(level_dir, explainer_results, kpi_results, initial_kpi, cfg)
        plot_feature_dropping(cfg, kpi_results, level_dir)

        all_results[target_vif] = {
            "sigma_d": sigma,
            "actual_max_vif": actual_vif,
            "rankings": {name: [f for f, _ in r["values"]] for name, r in explainer_results.items()},
            "slopes": {name: kpi["slope"] for name, kpi in kpi_results.items()},
            "slopes_std": {name: kpi.get("slope_std", 0.0) for name, kpi in kpi_results.items()},
            "slopes_per_seed": {name: kpi.get("slope_per_seed", [kpi["slope"]]) for name, kpi in kpi_results.items()},
            "timing": {name: r["mean_duration"] for name, r in explainer_results.items()},
        }

    methods = list(all_results[args.target_vifs[0]]["rankings"].keys())

    # --- Slopes summary ---
    print(f"\n{'='*60}")
    print("WEIGHTED DECREASING SLOPE (S, alpha=0.8)")
    print(f"{'='*60}")
    print(f"\n{'Method':<20}", end="")
    for t in args.target_vifs:
        print(f"  VIF={t:<6.0f}", end="")
    print()
    for m in methods:
        print(f"{m:<20}", end="")
        for t in args.target_vifs:
            print(f"  {all_results[t]['slopes'][m]:>8.3f}", end="")
        print()

    # --- Slopes plot ---
    fig, ax = plt.subplots(figsize=(10, 6))
    for m in methods:
        slopes = [all_results[t]["slopes"][m] for t in args.target_vifs]
        ax.plot(args.target_vifs, slopes, marker="o",
                label=DISPLAY_NAMES.get(m, m), linewidth=1.5, alpha=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("Target max VIF (log scale)", fontsize=14)
    ax.set_ylabel("Weighted Slope $S$", fontsize=14)
    ax.tick_params(labelsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=14, frameon=True, ncol=2)
    plt.tight_layout()
    for fmt in ["pdf"]:
        plt.savefig(output_base / f"slopes.{fmt}", format=fmt, dpi=300, bbox_inches="tight")
    plt.close()

    # --- Save summary ---
    summary = {
        "n_samples": args.n_samples,
        "n_features": n_features,
        "weights": WEIGHTS.tolist(),
        "primary_indices": PRIMARY_INDICES,
        "duplicate_map": DUPLICATE_MAP,
        "kpi": args.kpi,
        "target_vifs": args.target_vifs,
        "sigma_d_per_target": {str(t): s for t, s, _ in calib},
        "actual_vif_per_target": {str(t): a for t, _, a in calib},
        "slopes": {str(t): all_results[t]["slopes"] for t in args.target_vifs},
        "slopes_std": {str(t): all_results[t]["slopes_std"] for t in args.target_vifs},
        "slopes_per_seed": {str(t): all_results[t]["slopes_per_seed"] for t in args.target_vifs},
    }
    with open(output_base / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to {output_base}/")


if __name__ == "__main__":
    main()
