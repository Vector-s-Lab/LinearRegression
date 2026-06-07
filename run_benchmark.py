"""
Config-driven benchmark for XAI methods comparison.

Usage:
    python run_benchmark.py configs/abalone_r2.yaml
    python run_benchmark.py configs/abalone_r2.yaml --only exact_cis shap_explainer
    python run_benchmark.py configs/abalone_r2.yaml --resume results/previous_run/ --only exact_cis
    python run_benchmark.py --replot results/abalone_r2_20260401_200334
"""

import argparse
from datetime import datetime
from pathlib import Path

from benchmark import (
    build_char_func,
    build_model,
    build_shapg_custom_function,
    evaluate_feature_dropping,
    load_config,
    load_dataset,
    load_results,
    run_explainers,
    save_results,
)
from benchmark.plotting import plot_feature_dropping, plot_ranking_table, plot_timing


def replot(result_dir):
    """Regenerate plots from saved results without re-running experiments."""
    result_dir = Path(result_dir)
    print(f"Replotting from: {result_dir}")

    cfg, explainer_results, kpi_results, initial_kpi = load_results(result_dir)
    print(f"  Dataset: {cfg['dataset']}, KPI: {cfg['characteristic_function']}")
    print(f"  Methods: {list(explainer_results.keys())}")

    plot_feature_dropping(cfg, kpi_results, result_dir)
    plot_timing(cfg, explainer_results, result_dir)
    print("Done.")


def run(args):
    """Full benchmark run."""
    config_path = Path(args.config).resolve()
    cfg = load_config(config_path)

    # Filter explainers if --only is specified
    if args.only is not None:
        for name in list(cfg["explainers"].keys()):
            if name not in args.only:
                cfg["explainers"][name]["enabled"] = False

    # Output directory
    if args.output:
        output_dir = Path(args.output)
    elif args.resume:
        output_dir = Path(args.resume)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config_stem = config_path.stem
        output_dir = Path("results") / f"{config_stem}_{timestamp}"

    print(f"Config:  {config_path}")
    print(f"Output:  {output_dir}")

    # --- Load data ---
    print("\n[1/5] Loading dataset...")
    X, y = load_dataset(cfg)
    print(f"  {X.shape[0]} samples, {X.shape[1]} features")
    print(f"  Dataset: {cfg['dataset']}")
    print(f"  Characteristic function: {cfg['characteristic_function']}")

    # --- Build characteristic function ---
    print("\n[2/5] Building characteristic function...")
    char_func, X_train, X_test, y_train, y_test = build_char_func(cfg, X, y)
    shapg_custom_func = build_shapg_custom_function(cfg, X, y)

    # --- Resume previous results ---
    previous_results = {}
    if args.resume:
        print(f"\n  Resuming from {args.resume}")
        _, previous_results, _, _ = load_results(Path(args.resume))

    # --- Run explainers ---
    print("\n[3/5] Running explainers...")
    base_seed = cfg.get("seed", 42)
    n_runs = cfg.get("n_runs", 5)
    print(f"  Seed: {base_seed}, Runs: {n_runs}")
    new_results = run_explainers(
        cfg, X, y, char_func, X_train, X_test, y_train, y_test, shapg_custom_func
    )

    # Merge: new results override previous
    explainer_results = {**previous_results, **new_results}

    # --- Print runtime summary ---
    print("\n" + "=" * 60)
    print("Runtime Summary")
    print("=" * 60)
    print(f"{'Method':<25} {'Mean (s)':>10} {'Std (s)':>10} {'Runs':>6}")
    print("-" * 60)
    for name, r in explainer_results.items():
        print(f"{name:<25} {r['mean_duration']:>10.3f} {r['std_duration']:>10.3f} {r['n_runs']:>6}")
    print("=" * 60)

    # --- Feature ranking ---
    plot_ranking_table(cfg, explainer_results, output_dir)

    # --- Evaluate feature dropping ---
    print("\n[4/5] Evaluating feature dropping...")
    kpi_results, initial_kpi = evaluate_feature_dropping(cfg, X, y, explainer_results)

    print("\nWeighted Slopes:")
    beta = cfg["evaluation"]["weighted_slope_beta"]
    for name, kpi in kpi_results.items():
        print(f"  {name:<25} S={kpi['slope']:.6f}  (beta={beta})")

    # --- Save & plot ---
    print("\n[5/5] Saving results and generating plots...")
    save_results(output_dir, explainer_results, kpi_results, initial_kpi, cfg)
    plot_feature_dropping(cfg, kpi_results, output_dir)
    plot_timing(cfg, explainer_results, output_dir)


def main():
    parser = argparse.ArgumentParser(description="Run XAI benchmark from config.")
    parser.add_argument("config", type=str, nargs="?", help="Path to YAML config file")
    parser.add_argument("--output", type=str, default=None, help="Override output directory")
    parser.add_argument(
        "--only", nargs="*", default=None,
        help="Only run these explainer names (e.g. --only exact_cis shap_explainer)"
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Load previous results from this directory, only re-run --only methods"
    )
    parser.add_argument(
        "--replot", type=str, nargs="+", default=None,
        help="Replot from saved results dir(s) (e.g. --replot results/abalone_r2_*)"
    )
    args = parser.parse_args()

    if args.replot:
        for d in args.replot:
            replot(d)
        return

    if not args.config:
        parser.error("config is required when not using --replot")

    run(args)


if __name__ == "__main__":
    main()
