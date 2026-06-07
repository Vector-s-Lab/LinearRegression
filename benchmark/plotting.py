"""Plot generation for XAI comparison."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Optional plotting style
try:
    import gnuplot_style as gp

    gp.use("all")
except ImportError:
    pass


DISPLAY_NAMES = {
    "shap_explainer": "SHAP Explainer",
    "sampling_shap": "Sampling SHAP",
    "kernel_shap": "Kernel SHAP",
    "original_shapg": "Original ShapG",
    "improved_shapg": "Improved ShapG",
    "exact_shapley": "Exact Shapley",
    "exact_cis": "Exact CIS",
}

KPI_LABELS = {
    "r2": "$R^2$",
    "adj_r2": "$R^2_{adj}$",
    "aic": "1/AIC",
    "f_statistic": "F-statistic",
}


def _decimals_for_total_digits(x, total_digits=4):
    """Number of decimal places so that integer + decimal digits equals total_digits.

    Matches the formatting used by the LaTeX tables (timing_all, slopes_all)
    so that figure legends and tables share the same numeric style.
    """
    if x == 0 or abs(x) < 1:
        int_digits = 1
    else:
        int_digits = len(str(int(abs(x))))
    return max(0, total_digits - int_digits)


def _slope_total_digits(kpi, kind="mean"):
    """Per-KPI total-digit budget for slope labels in figure legends.

    Mirrors ``generate_paper_assets._slope_decimals`` so figure legends and
    LaTeX tables show identical numeric formatting.

    +------------------+----------+----------+
    | KPI              | mean     | std      |
    +------------------+----------+----------+
    | r2 / adj_r2      | 4 total  | 4 total  |
    | aic              | 6 total  | 2 total  |
    | f_statistic / fstat | 3 total | 3 total |
    +------------------+----------+----------+
    """
    if kpi == "aic":
        return 6 if kind == "mean" else 2
    if kpi in ("f_statistic", "fstat"):
        return 3
    return 4  # r2, adj_r2


def plot_feature_dropping(cfg, kpi_results, output_dir):
    """Plot feature-dropping comparison curves."""
    plot_cfg = cfg.get("plot", {})
    figsize = tuple(plot_cfg.get("figsize", [10, 6]))
    dpi = plot_cfg.get("dpi", 300)
    formats = plot_cfg.get("format", ["pdf"])
    kpi_type = cfg["characteristic_function"]
    limit = cfg["evaluation"]["drop_limit"]
    beta = cfg["evaluation"]["weighted_slope_beta"]
    dataset = cfg["dataset"]

    output_dir = Path(output_dir)

    plt.figure(figsize=figsize)

    for name, kpi in kpi_results.items():
        metrics = kpi["metrics"]
        slope = kpi["slope"]
        slope_std = kpi.get("slope_std", 0.0)
        display = DISPLAY_NAMES.get(name, name)
        d_mean = _decimals_for_total_digits(slope, _slope_total_digits(kpi_type, "mean"))
        d_std = _decimals_for_total_digits(slope_std, _slope_total_digits(kpi_type, "std"))
        label = (
            f"{display} (S={slope:.{d_mean}f} "
            f"$\\pm$ {slope_std:.{d_std}f})"
        )
        plt.plot(
            range(len(metrics)), metrics,
            label=label, linewidth=1.5, alpha=0.7,
        )

    kpi_label = KPI_LABELS.get(kpi_type, kpi_type)
    plt.xlabel("Number of Features Dropped", fontsize=14)
    plt.ylabel(kpi_label, fontsize=14)
    plt.xticks(range(limit), fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best", fontsize=14, frameon=True, ncol=1)
    plt.tight_layout()

    for fmt in formats:
        outfile = output_dir / f"feature_dropping.{fmt}"
        plt.savefig(outfile, format=fmt, dpi=dpi, bbox_inches="tight")
        print(f"  Plot saved: {outfile}")
    plt.close()


def plot_timing(cfg, explainer_results, output_dir):
    """Plot horizontal bar chart of computation times with error bars."""
    plot_cfg = cfg.get("plot", {})
    dpi = plot_cfg.get("dpi", 300)
    formats = plot_cfg.get("format", ["pdf"])
    output_dir = Path(output_dir)

    names = list(explainer_results.keys())
    means = [explainer_results[n]["mean_duration"] for n in names]
    stds = [explainer_results[n]["std_duration"] for n in names]
    display_names = [DISPLAY_NAMES.get(n, n) for n in names]

    fig, ax = plt.subplots(figsize=(8, max(3, len(names) * 0.6)))
    y_pos = range(len(names))
    ax.barh(y_pos, means, xerr=stds, alpha=0.7, capsize=3)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(display_names)
    ax.set_xlabel("Time (seconds)")
    ax.set_title("Computation Time Comparison")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()

    for fmt in formats:
        outfile = output_dir / f"timing.{fmt}"
        plt.savefig(outfile, format=fmt, dpi=dpi, bbox_inches="tight")
    plt.close()


def plot_ranking_table(cfg, explainer_results, output_dir):
    """Print and return feature ranking comparison DataFrame."""
    methods = {}
    for name, r in explainer_results.items():
        display = DISPLAY_NAMES.get(name, name)
        methods[display] = [f for f, _ in r["values"]]

    df = pd.DataFrame(methods)
    df.insert(0, "Rank", [f"Top{i+1}" for i in range(len(df))])

    print("\nFeature Ranking Comparison:")
    print(df.to_string(index=False))
    return df
