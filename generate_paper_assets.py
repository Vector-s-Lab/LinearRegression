"""
Generate paper assets from experiment results.

1. Copy feature-dropping and timing figures from results/ to paper/
2. Copy multicollinearity figures (accuracy, stability, slopes) from results/ to paper/
3. Generate LaTeX tables (rankings, timing, slopes, multicollinearity) into paper/tables/

Usage:
    python generate_paper_assets.py
    python generate_paper_assets.py --timestamp 20260415_153011
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from benchmark.data import load_dataset
from benchmark.statistical_methods import STATISTICAL_METHODS, compute_all_statistical_rankings


RESULTS_DIR = Path("results")
FIGURES_DIR = Path("paper")  # Figures live next to sn-article.tex (no subfolder, no dots in filenames)
TABLES_DIR = Path("paper/tables")

DATASETS = ["abalone", "admission", "concrete", "ucs_scm"]
KPIS = ["r2", "adj_r2", "aic", "fstat"]

DATASET_DISPLAY = {
    "abalone": "Abalone",
    "admission": "Admission",
    "concrete": "Concrete",
    "ucs_scm": "UCS-SCM",
}

KPI_DISPLAY = {
    "r2": "$R^2$",
    "adj_r2": "$R^2_{adj}$",
    "aic": "1/AIC",
    "fstat": "F-statistic",
}

# For multicollinearity results, kpi names differ (no "fstat" alias)
KPI_MULTI_MAP = {
    "r2": "r2",
    "aic": "aic",
    "f_statistic": "f_statistic",
}

KPI_MULTI_DISPLAY = {
    "r2": "$R^2$",
    "adj_r2": "$R^2_{adj}$",
    "aic": "1/AIC",
    "f_statistic": "F-statistic",
}

SNR_DISPLAY = {
    1.0: "Noisy ($R^2 \\approx 0.50$)",
    2.38: "Moderate ($R^2 \\approx 0.85$)",
    4.36: "Well-fitted ($R^2 \\approx 0.95$)",
}

METHOD_DISPLAY = {
    "exact_cis": "Exact CIS",
    "exact_shapley": "Exact Shapley",
    "shap_explainer": "SHAP Explainer",
    "sampling_shap": "Sampling SHAP",
    "kernel_shap": "Kernel SHAP",
    "original_shapg": "Original ShapG",
    "improved_shapg": "Improved ShapG",
}


def find_result_dir(dataset, kpi, timestamp=None):
    """Find the result directory for a dataset/kpi combination."""
    pattern = f"{dataset}_{kpi}_*"
    matches = sorted(RESULTS_DIR.glob(pattern))
    if timestamp:
        matches = [m for m in matches if timestamp in m.name]
    # Filter to dirs with actual results (not just run.log)
    matches = [m for m in matches if (m / "kpi.json").exists()]
    if not matches:
        return None
    return matches[-1]


def find_multicollinearity_dirs():
    """Find all multicollinearity result dirs, grouped by (snr, kpi)."""
    dirs = {}
    for d in sorted(RESULTS_DIR.glob("multicollinearity_*")):
        summary_path = d / "summary.json"
        if not summary_path.exists():
            continue
        summary = json.load(open(summary_path))
        snr = summary.get("snr")
        kpi = summary.get("kpi")
        if snr is not None and kpi is not None:
            dirs[(snr, kpi)] = d
    return dirs


# ============================================================
# Copy figures into paper/
# ============================================================

def copy_figures(timestamp=None):
    """Copy feature-dropping and timing PDF figures to paper/."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    copied = 0
    for ds in DATASETS:
        for kpi in KPIS:
            result_dir = find_result_dir(ds, kpi, timestamp)
            if result_dir is None:
                continue
            for fig_type in ["feature_dropping", "timing"]:
                src = result_dir / f"{fig_type}.pdf"
                if src.exists():
                    dst = FIGURES_DIR / f"{ds}_{kpi}_{fig_type}.pdf"
                    shutil.copy2(src, dst)
                    copied += 1
    print(f"  Copied {copied} main-benchmark figures")


def copy_multicollinearity_figures():
    """Copy accuracy, stability, slopes PDF figures from multicollinearity results."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    dirs = find_multicollinearity_dirs()
    copied = 0
    for (snr, kpi), d in dirs.items():
        for fig_type in ["accuracy", "stability", "slopes"]:
            src = d / f"{fig_type}.pdf"
            if src.exists():
                # Replace dots in SNR (e.g. 1.0 -> 1p0) so the filename is dot-free.
                snr_tag = str(snr).replace(".", "p")
                dst = FIGURES_DIR / f"multicollinearity_snr{snr_tag}_{kpi}_{fig_type}.pdf"
                shutil.copy2(src, dst)
                copied += 1
    print(f"  Copied {copied} multicollinearity figures")


def copy_near_dependency_figures():
    """Copy slopes PDF figures from near-dependency runs into paper/.

    Outputs ``near_dependency_<kpi>_slopes.pdf`` for each KPI, picking the
    latest timestamped directory per KPI."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    dirs = find_near_dependency_dirs()
    copied = 0
    for kpi, d in dirs.items():
        src = d / "slopes.pdf"
        if src.exists():
            dst = FIGURES_DIR / f"near_dependency_{kpi}_slopes.pdf"
            shutil.copy2(src, dst)
            copied += 1
    print(f"  Copied {copied} near-dependency figures")


# ============================================================
# Main-benchmark LaTeX tables
# ============================================================

def _merge_identical_rankings(columns, unmergeable=()):
    """Merge any columns with identical rankings, even if non-adjacent.

    columns: list of (display_label, ranking_list) pairs, in display order.
    unmergeable: iterable of display labels that must remain in their own
        column even if their ranking matches another method's.
    Returns: list of (compound_display_label, ranking_list) pairs. Compound
    labels use \\makecell[l]{...} (left-aligned stacked names) to keep the
    table compact.
    """
    unmergeable = set(unmergeable)
    groups = []  # list of dicts: {'ranking': [...], 'labels': [...], 'fixed': bool}
    for label, ranking in columns:
        matched = False
        if label not in unmergeable:
            for g in groups:
                if g['fixed']:
                    continue
                if g['ranking'] == ranking:
                    g['labels'].append(label)
                    matched = True
                    break
        if not matched:
            groups.append({
                'ranking': ranking,
                'labels': [label],
                'fixed': label in unmergeable,
            })

    merged = []
    for g in groups:
        if len(g['labels']) == 1:
            merged.append((g['labels'][0], g['ranking']))
        else:
            compound = "\\makecell[l]{" + "\\\\".join(g['labels']) + "}"
            merged.append((compound, g['ranking']))
    return merged


def _collect_rankings_all_kpi(dataset, timestamp=None):
    """Return {method: {kpi: tuple_of_features}} for all available methods/KPIs."""
    data = {}
    for kpi in KPIS:
        result_dir = find_result_dir(dataset, kpi, timestamp)
        if result_dir is None:
            continue
        values = json.load(open(result_dir / "feature_values.json"))
        for method, feat_vals in values.items():
            ranking = tuple(
                f for f, _ in sorted(feat_vals.items(), key=lambda x: x[1], reverse=True)
            )
            data.setdefault(method, {})[kpi] = ranking
    return data


def _escape_feat(name):
    return str(name).replace("_", r"\_\allowbreak ")


def generate_ranking_table(dataset, timestamp=None):
    """Generate LaTeX table environment with rankings for one dataset.

    The table has 4 KPI blocks (R^2, R^2_{adj}, 1/AIC, F-statistic), each block
    listing the feature rankings in order of decreasing importance. Methods whose
    full 4-tuple of per-KPI rankings is identical are merged into a single column
    (with compound \\makecell[l]{...} label), so the column count reflects the
    number of distinct ranking "personalities" on this dataset.
    """
    data = _collect_rankings_all_kpi(dataset, timestamp)
    if not data:
        return ""

    methods = [m for m in METHOD_DISPLAY if m in data]
    ds_display = DATASET_DISPLAY.get(dataset, dataset)
    n_features = max(len(r) for d in data.values() for r in d.values())

    # Group methods by the full (r2, adj_r2, aic, fstat) ranking tuple. Two
    # methods are merged iff they produce identical rankings for every KPI.
    group_map = {}  # key -> {"labels": [...], "per_kpi": {kpi: ranking}}
    order = []  # preserve first-seen order
    for m in methods:
        by_kpi = data[m]
        key = tuple(by_kpi.get(k) for k in KPIS)
        if key not in group_map:
            group_map[key] = {"labels": [], "per_kpi": by_kpi}
            order.append(key)
        group_map[key]["labels"].append(METHOD_DISPLAY[m])

    headers = []
    for key in order:
        labels = group_map[key]["labels"]
        if len(labels) == 1:
            headers.append(labels[0])
        else:
            headers.append("\\makecell[l]{" + "\\\\".join(labels) + "}")

    n_cols = len(headers) + 1  # +1 for Rank

    lines = []
    lines.append("\\begin{table}[htb]")
    lines.append("\\centering")
    lines.append("\\tiny")
    lines.append(
        f"\\caption{{Feature rankings by XAI methods for ``{ds_display}'' dataset "
        f"under all four characteristic functions ($R^2$, $R^2_{{adj}}$, $1/AIC$, "
        f"$F$-statistic). Methods whose rankings are identical across all four "
        f"KPIs are merged into a single column.}}"
    )
    lines.append(f"\\label{{tab:ranking_{dataset}}}")
    lines.append("\\setlength{\\tabcolsep}{2pt}")
    lines.append(
        "\\begin{tabularx}{\\textwidth}{l*{"
        + str(len(headers))
        + "}{>{\\raggedright\\arraybackslash}X}}"
    )
    lines.append("\\toprule")
    lines.append("Rank & " + " & ".join(headers) + " \\\\")

    for kpi in KPIS:
        lines.append("\\midrule")
        lines.append(
            f"\\multicolumn{{{n_cols}}}{{c}}{{\\textit{{KPI: {KPI_DISPLAY[kpi]}}}}} \\\\"
        )
        lines.append("\\midrule")
        for i in range(n_features):
            row = [f"\\nth{{{i+1}}}"]
            for key in order:
                ranking = group_map[key]["per_kpi"].get(kpi, ())
                feat = ranking[i] if i < len(ranking) else ""
                row.append(_escape_feat(feat))
            lines.append(" & ".join(row) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabularx}")
    lines.append("\\end{table}")
    return "\n".join(lines)


def generate_ranking_stability_table(timestamp=None):
    """One compact summary table: # distinct rankings per (dataset, method) across the 4 KPIs.

    A value of 1 means the method produces the same ranking regardless of which
    characteristic function is used; 4 means every KPI yields a different ranking.
    Lower is more KPI-stable.
    """
    rows = []
    for ds in DATASETS:
        data = _collect_rankings_all_kpi(ds, timestamp)
        if not data:
            continue
        row = {"Dataset": DATASET_DISPLAY.get(ds, ds)}
        for method in METHOD_DISPLAY:
            if method not in data:
                row[METHOD_DISPLAY[method]] = "---"
                continue
            distinct = len({data[method].get(k) for k in KPIS if k in data[method]})
            row[METHOD_DISPLAY[method]] = str(distinct)
        rows.append(row)

    if not rows:
        return ""
    df = pd.DataFrame(rows)
    method_cols = [c for c in df.columns if c != "Dataset"]
    df = _bold_best_per_row(df, method_cols, how="min")

    lines = []
    lines.append("\\begin{table}[htb]")
    lines.append("\\centering")
    lines.append("\\scriptsize")
    lines.append(
        "\\caption{Number of distinct feature rankings each XAI method produces "
        "across the four characteristic functions ($R^2$, $R^2_{adj}$, $1/AIC$, "
        "$F$-statistic). Lower is more KPI-stable (1 = same ranking under all four "
        "KPIs; 4 = a different ranking under each KPI). Most stable method per "
        "dataset in \\textbf{bold}.}"
    )
    lines.append("\\label{tab:ranking_stability}")
    lines.append("\\setlength{\\tabcolsep}{4pt}")
    n_methods = len(method_cols)
    lines.append(
        f"\\begin{{tabularx}}{{\\textwidth}}{{l*{{{n_methods}}}{{>{{\\centering\\arraybackslash}}X}}}}"
    )
    lines.append("\\toprule")
    lines.append("Dataset & " + " & ".join(method_cols) + " \\\\")
    lines.append("\\midrule")
    for _, r in df.iterrows():
        lines.append(
            r["Dataset"] + " & " + " & ".join(str(r[c]) for c in method_cols) + " \\\\"
        )
    lines.append("\\bottomrule")
    lines.append("\\end{tabularx}")
    lines.append("\\end{table}")
    return "\n".join(lines)


def _to_tabularx_wide(df):
    """Convert df.to_latex() output to tabularx with right-aligned X columns for the trailing
    numeric columns. Assumes first two columns (Dataset, KPI) are left-aligned text."""
    tabular = df.to_latex(index=False, escape=False).strip()
    n_x = len(df.columns) - 2
    new_open = (
        f"\\begin{{tabularx}}{{\\textwidth}}"
        f"{{ll*{{{n_x}}}{{>{{\\raggedleft\\arraybackslash}}X}}}}"
    )
    # Replace the opening \begin{tabular}{...} and closing \end{tabular}.
    first_nl = tabular.index("\n")
    tabular = new_open + tabular[first_nl:]
    tabular = tabular.replace("\\end{tabular}", "\\end{tabularx}")
    return tabular


def _bold_best_per_row(df, value_cols, how="max"):
    """Return a new DataFrame with the best numeric value(s) per row wrapped in \\textbf{}.
    All columns tied at the best value are bolded.

    value_cols: columns to consider (string-formatted numbers). Non-value columns (e.g. labels) are left alone.
    how: "max" for higher-is-better, "min" for lower-is-better.
    Formatted strings like "0.123 $\\pm$ 0.004" are parsed on their first number.
    """
    import re
    df = df.copy()
    for idx, row in df.iterrows():
        nums = {}
        for col in value_cols:
            v = row[col]
            if not isinstance(v, str) or v.strip() == "---":
                continue
            m = re.match(r"\s*(-?\d+(?:\.\d+)?)", v)
            if not m:
                continue
            nums[col] = float(m.group(1))
        if not nums:
            continue
        best = max(nums.values()) if how == "max" else min(nums.values())
        for col, v in nums.items():
            if v == best:
                df.at[idx, col] = "\\textbf{" + df.at[idx, col] + "}"
    return df


def generate_statistical_ranking_table(dataset):
    """Generate LaTeX table of rankings given by classical statistical methods."""
    X, y = load_dataset({"dataset": dataset})
    rankings = compute_all_statistical_rankings(X, y)

    method_order = ["beta", "squared_corr", "pratt", "structure",
                    "partial", "semipartial", "dominance"]
    n_features = X.shape[1]
    ds_display = DATASET_DISPLAY.get(dataset, dataset)

    # Collect (display_label, ranking-as-feature-list) pairs, then merge identical ones.
    # Keep "Dominance" in its own column even if it matches another method's ranking.
    columns = [
        (STATISTICAL_METHODS[m][0], [f for f, _ in rankings[m]])
        for m in method_order
    ]
    dominance_label = STATISTICAL_METHODS["dominance"][0]
    merged = _merge_identical_rankings(columns, unmergeable=[dominance_label])

    lines = []
    lines.append("\\begin{table}[htb]")
    lines.append("\\centering")
    lines.append("\\tiny")
    lines.append(f"\\caption{{Feature rankings given by classical statistical methods for ``{ds_display}'' dataset. Methods producing identical rankings are merged into a single column.}}")
    lines.append(f"\\label{{tab:statistical_{dataset}}}")
    lines.append("\\setlength{\\tabcolsep}{2pt}")
    lines.append(
        "\\begin{tabularx}{\\textwidth}{l*{"
        + str(len(merged))
        + "}{>{\\raggedright\\arraybackslash}X}}"
    )
    lines.append("\\toprule")
    lines.append("Rank & " + " & ".join(label for label, _ in merged) + " \\\\")
    lines.append("\\midrule")

    for i in range(n_features):
        row = [f"\\nth{{{i+1}}}"]
        for _, ranking in merged:
            feat = ranking[i] if i < len(ranking) else ""
            row.append(str(feat).replace("_", r"\_\allowbreak "))
        lines.append(" & ".join(row) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabularx}")
    lines.append("\\end{table}")
    return "\n".join(lines)


def _render_grouped_by_dataset(dataset_blocks, caption, label, col_header="KPI"):
    """Render a table where rows are grouped by dataset. Each block supplies a label
    (first-column value per row) and the method cells."""
    method_cols = [METHOD_DISPLAY[m] for m in METHOD_DISPLAY]
    n_cols = 1 + len(method_cols)
    col_spec = f"l*{{{len(method_cols)}}}{{>{{\\raggedright\\arraybackslash}}X}}"

    lines = []
    lines.append("\\begin{table}[htb]")
    lines.append("\\centering")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append("\\tiny")
    lines.append("\\setlength{\\tabcolsep}{2pt}")
    lines.append(f"\\begin{{tabularx}}{{\\textwidth}}{{{col_spec}}}")
    lines.append("\\toprule")
    header = [col_header] + method_cols
    lines.append(" & ".join(header) + " \\\\")

    for ds_display, df in dataset_blocks:
        lines.append("\\midrule")
        lines.append(
            f"\\multicolumn{{{n_cols}}}{{c}}{{\\textit{{Dataset: {ds_display}}}}} \\\\"
        )
        lines.append("\\midrule")
        for _, row in df.iterrows():
            cells = [row[col_header]] + [row[c] for c in method_cols]
            lines.append(" & ".join(cells) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabularx}")
    lines.append("\\end{table}")
    return "\n".join(lines)


def _fixed_digits_decimals(x, total_digits=4):
    """Return the number of decimal places to keep so that
    (integer digits) + (decimal digits) == total_digits. For |x| >= 10^total_digits the
    returned value is 0."""
    if x == 0 or abs(x) < 1:
        int_digits = 1
    else:
        int_digits = len(str(int(abs(x))))
    return max(0, total_digits - int_digits)


# Std-font commands used by the slope/timing table formatters.
# Defined here so generate_timing_table can reference _STD_FONT_TINY_TABLE.
_STD_FONT_TINY_TABLE = "\\fontsize{4pt}{4.8pt}\\selectfont"
_STD_FONT_SCRIPTSIZE_TABLE = "\\tiny"


def generate_timing_table(timestamp=None):
    """Generate LaTeX table for timing, grouped by dataset. Fastest per row in bold.
    Values are formatted with 4 total digits (e.g. 338.3, 28.87, 1.581, 0.006); std uses
    the same number of decimals as the mean for visual consistency."""
    method_cols = [METHOD_DISPLAY[m] for m in METHOD_DISPLAY]
    dataset_blocks = []
    for ds in DATASETS:
        rows = []
        for kpi in KPIS:
            result_dir = find_result_dir(ds, kpi, timestamp)
            if result_dir is None:
                continue
            timing = json.load(open(result_dir / "timing.json"))
            row = {"KPI": KPI_DISPLAY.get(kpi, kpi)}
            for method in METHOD_DISPLAY:
                if method in timing:
                    t = timing[method]
                    d_mean = _fixed_digits_decimals(t["mean"])
                    if t["n_runs"] > 1:
                        d_std = _fixed_digits_decimals(t["std"])
                        row[METHOD_DISPLAY[method]] = (
                            f"{t['mean']:.{d_mean}f} "
                            f"{{{_STD_FONT_TINY_TABLE} $\\pm$ {t['std']:.{d_std}f}}}"
                        )
                    else:
                        row[METHOD_DISPLAY[method]] = f"{t['mean']:.{d_mean}f}"
                else:
                    row[METHOD_DISPLAY[method]] = "---"
            rows.append(row)
        if rows:
            df = pd.DataFrame(rows)
            df = _bold_best_per_row(df, method_cols, how="min")
            dataset_blocks.append((DATASET_DISPLAY.get(ds, ds), df))

    if not dataset_blocks:
        return ""
    return _render_grouped_by_dataset(
        dataset_blocks,
        caption="Running time (seconds, mean $\\pm$ std) of XAI methods across all datasets and characteristic functions. Fastest per row in \\textbf{bold}.",
        label="tab:timing_all",
    )


def _slope_decimals(kpi, val, kind="mean"):
    """Per-KPI total-digit budget (int + dec) for slope-table cells.

    +------------------+----------+----------+
    | KPI              | mean     | std      |
    +------------------+----------+----------+
    | R^2 / R^2_adj    | 4 total  | 4 total  |
    | 1/AIC            | 6 total  | 2 total  |
    | F-statistic      | 3 total  | 3 total  |
    +------------------+----------+----------+

    Accepts either the short key (``fstat``) used in main-benchmark configs
    or the full key (``f_statistic``) used in multicollinearity / near-dep
    summary files.
    """
    if kpi == "aic":
        total = 6 if kind == "mean" else 2
    elif kpi in ("f_statistic", "fstat"):
        total = 3
    else:  # r2 / adj_r2
        total = 4
    return _fixed_digits_decimals(val, total_digits=total)


def _format_mean_std(mean, std, kpi=None, std_font=_STD_FONT_TINY_TABLE):
    """Format ``mean $\\pm$ std`` for slope tables, with decimal places chosen
    per-KPI when ``kpi`` is provided. The std is wrapped in a smaller font so
    it visually de-emphasises relative to the mean.

    ``std_font`` controls which LaTeX size command wraps the std. Use
    ``_STD_FONT_TINY_TABLE`` (default) for tables whose body is ``\\tiny``
    (e.g. ``slopes_all``, ``timing_all``); use ``_STD_FONT_SCRIPTSIZE_TABLE``
    for tables whose body is ``\\scriptsize`` (e.g. ``multi_slopes``,
    ``near_dep_slopes``)."""
    if mean is None:
        return "---"
    if kpi is None:
        # Fallback: 4-total-digit auto behaviour (timing-style).
        d_mean = _fixed_digits_decimals(mean)
        d_std = _fixed_digits_decimals(std)
    else:
        d_mean = _slope_decimals(kpi, mean, kind="mean")
        d_std = _slope_decimals(kpi, std, kind="std")
    std_fmt = f"{{{std_font} $\\pm$ {std:.{d_std}f}}}"
    return f"{mean:.{d_mean}f} {std_fmt}"


def generate_slope_table(timestamp=None):
    """Generate LaTeX table for weighted slopes (mean $\\pm$ std over seeds), grouped by dataset.
    Best per row in bold."""
    method_cols = [METHOD_DISPLAY[m] for m in METHOD_DISPLAY]
    dataset_blocks = []
    for ds in DATASETS:
        rows = []
        for kpi in KPIS:
            result_dir = find_result_dir(ds, kpi, timestamp)
            if result_dir is None:
                continue
            kpi_data = json.load(open(result_dir / "kpi.json"))
            row = {"KPI": KPI_DISPLAY.get(kpi, kpi)}
            for method in METHOD_DISPLAY:
                if method in kpi_data:
                    mean = kpi_data[method]["slope"]
                    std = kpi_data[method].get("slope_std", 0.0)
                    row[METHOD_DISPLAY[method]] = _format_mean_std(mean, std, kpi=kpi)
                else:
                    row[METHOD_DISPLAY[method]] = "---"
            rows.append(row)
        if rows:
            df = pd.DataFrame(rows)
            df = _bold_best_per_row(df, method_cols, how="max")
            dataset_blocks.append((DATASET_DISPLAY.get(ds, ds), df))

    if not dataset_blocks:
        return ""
    return _render_grouped_by_dataset(
        dataset_blocks,
        caption="Weighted decreasing slopes ($S$, $\\alpha=0.8$) of XAI methods across all datasets and characteristic functions, reported as mean $\\pm$ standard deviation over 10 seeds. Higher $S$ indicates better feature ranking. Best per row in \\textbf{bold}.",
        label="tab:slopes_all",
    )


# ============================================================
# Multicollinearity LaTeX tables
# ============================================================

def generate_multicollinearity_accuracy_table():
    """Generate LaTeX table environment: Kendall tau vs ground truth across SNR and correlation levels."""
    dirs = find_multicollinearity_dirs()
    if not dirs:
        return ""

    all_rows = []
    for (snr, kpi), d in sorted(dirs.items()):
        summary = json.load(open(d / "summary.json"))
        accuracy = summary.get("accuracy", {})
        levels = summary.get("levels", [])
        snr_label = SNR_DISPLAY.get(snr, f"SNR={snr}")
        kpi_label = KPI_MULTI_DISPLAY.get(kpi, kpi)

        for method in METHOD_DISPLAY:
            if method not in accuracy:
                continue
            row = {
                "SNR": snr_label,
                "KPI": kpi_label,
                "Method": METHOD_DISPLAY[method],
            }
            for level in levels:
                tau = accuracy[method].get(str(level))
                if tau is not None:
                    row[f"$\\rho={level}$"] = f"{tau:.3f}"
                else:
                    row[f"$\\rho={level}$"] = "---"
            all_rows.append(row)

    if not all_rows:
        return ""
    df = pd.DataFrame(all_rows)
    # Within each (SNR, KPI) group, for each rho column, bold ALL methods tied at the best
    rho_cols = [c for c in df.columns if c.startswith("$\\rho")]
    import re
    for (snr_label, kpi_label), group in df.groupby(["SNR", "KPI"]):
        for col in rho_cols:
            vals = {}
            for idx in group.index:
                v = df.at[idx, col]
                m = re.match(r"\s*(-?\d+(?:\.\d+)?)", v)
                if m:
                    vals[idx] = float(m.group(1))
            if vals:
                best = max(vals.values())
                for idx, v in vals.items():
                    if v == best:
                        df.at[idx, col] = "\\textbf{" + df.at[idx, col] + "}"
    tabular = df.to_latex(index=False, escape=False)

    lines = []
    lines.append("\\begin{table}[htb]")
    lines.append("\\centering")
    lines.append("\\caption{Ranking accuracy (Kendall $\\tau$ vs ground truth) under varying multicollinearity levels and signal-to-noise ratios. Best method per (SNR, KPI, $\\rho$) combination in \\textbf{bold}.}")
    lines.append("\\label{tab:multi_accuracy}")
    lines.append("\\scriptsize")
    lines.append("\\setlength{\\tabcolsep}{4pt}")
    lines.append(tabular.strip())
    lines.append("\\end{table}")
    return "\n".join(lines)


def _bold_best_per_column_within_group(df, group_col, value_cols, how="max"):
    """Bold all values tied at the best within each group × column."""
    import re
    for grp_val in df[group_col].unique():
        mask = df[group_col] == grp_val
        sub_idx = df[mask].index
        for col in value_cols:
            vals = {}
            for idx in sub_idx:
                v = df.at[idx, col]
                m = re.match(r"\s*(-?\d+(?:\.\d+)?)", v)
                if m:
                    vals[idx] = float(m.group(1))
            if vals:
                best = max(vals.values()) if how == "max" else min(vals.values())
                for idx, v in vals.items():
                    if v == best:
                        df.at[idx, col] = "\\textbf{" + df.at[idx, col] + "}"
    return df


def _render_grouped_by_snr(df, caption, label, size="\\scriptsize", placement="htb"):
    """Render a table where rows are grouped by SNR (first column), with an italic section
    header per SNR block. The df must have columns [SNR, Method, <rho cols...>]."""
    rho_cols = [c for c in df.columns if c.startswith("$\\rho")]
    n_cols = 1 + len(rho_cols)
    col_spec = "l" + "l" * len(rho_cols)

    lines = []
    lines.append(f"\\begin{{table}}[{placement}]")
    lines.append("\\centering")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append(size)
    lines.append("\\setlength{\\tabcolsep}{4pt}")
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append("\\toprule")
    header = ["Method"] + rho_cols
    lines.append(" & ".join(header) + " \\\\")

    for snr_label in df["SNR"].drop_duplicates():
        sub = df[df["SNR"] == snr_label]
        lines.append("\\midrule")
        lines.append(
            f"\\multicolumn{{{n_cols}}}{{c}}{{\\textit{{SNR: {snr_label}}}}} \\\\"
        )
        lines.append("\\midrule")
        for _, row in sub.iterrows():
            cells = [row["Method"]] + [row[c] for c in rho_cols]
            lines.append(" & ".join(cells) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines)


def _build_multicollinearity_slopes_df(target_kpi):
    """Collect a flat df [SNR, Method, rho cols...] with bolded best-per-column-within-SNR."""
    dirs = find_multicollinearity_dirs()
    if not dirs:
        return None

    all_rows = []
    for (snr, kpi) in sorted(dirs.keys()):
        if kpi != target_kpi:
            continue
        d = dirs[(snr, kpi)]
        summary = json.load(open(d / "summary.json"))
        slopes = summary.get("slopes", {})
        slopes_std = summary.get("slopes_std", {})
        levels = summary.get("levels", [])
        snr_label = SNR_DISPLAY.get(snr, f"SNR={snr}")

        for method in METHOD_DISPLAY:
            if not any(method in slopes.get(str(level), {}) for level in levels):
                continue
            row = {"SNR": snr_label, "Method": METHOD_DISPLAY[method]}
            for level in levels:
                s = slopes.get(str(level), {}).get(method)
                ss = slopes_std.get(str(level), {}).get(method, 0.0) if slopes_std else 0.0
                row[f"$\\rho={level}$"] = (
                    _format_mean_std(s, ss, kpi=kpi, std_font=_STD_FONT_SCRIPTSIZE_TABLE)
                    if s is not None else "---"
                )
            all_rows.append(row)

    if not all_rows:
        return None
    df = pd.DataFrame(all_rows)
    rho_cols = [c for c in df.columns if c.startswith("$\\rho")]
    return _bold_best_per_column_within_group(df, "SNR", rho_cols, how="max")


def generate_multicollinearity_slopes_table():
    """Generate LaTeX table: weighted slope S across SNR and rho levels ($R^2$)."""
    df = _build_multicollinearity_slopes_df("r2")
    if df is None:
        return ""
    return _render_grouped_by_snr(
        df,
        caption=(
            "Weighted decreasing slopes ($S$, $\\alpha=0.8$, $R^2$ characteristic function) "
            "under varying multicollinearity and signal-to-noise ratios, reported as mean "
            "$\\pm$ standard deviation over 10 seeds. Higher $S$ indicates better feature "
            "ranking. Best method per (SNR, $\\rho$) combination in \\textbf{bold}."
        ),
        label="tab:multi_slopes",
        size="\\scriptsize",
    )


def _generate_multicollinearity_slopes_for_kpi(target_kpi):
    """Generate a single multicollinearity slope table for one specified KPI."""
    df = _build_multicollinearity_slopes_df(target_kpi)
    if df is None:
        return ""
    kpi_label = KPI_MULTI_DISPLAY.get(target_kpi, target_kpi)
    return _render_grouped_by_snr(
        df,
        caption=(
            f"Weighted decreasing slopes ($S$, $\\alpha=0.8$) under multicollinearity "
            f"with {kpi_label} as the characteristic function, reported as mean "
            f"$\\pm$ standard deviation over 10 seeds. Best method per "
            f"(SNR, $\\rho$) combination in \\textbf{{bold}}."
        ),
        label=f"tab:multi_slopes_{target_kpi}",
        size="\\tiny",
        placement="htb",
    )


def generate_multicollinearity_slopes_adj_r2_table():
    return _generate_multicollinearity_slopes_for_kpi("adj_r2")


def generate_multicollinearity_slopes_aic_table():
    return _generate_multicollinearity_slopes_for_kpi("aic")


def generate_multicollinearity_slopes_fstat_table():
    return _generate_multicollinearity_slopes_for_kpi("f_statistic")


def generate_multicollinearity_stability_table():
    """Generate LaTeX table: Kendall tau ranking stability vs rho=0 baseline across SNR levels."""
    dirs = find_multicollinearity_dirs()
    if not dirs:
        return ""
    r2_dirs = {snr: d for (snr, kpi), d in dirs.items() if kpi == "r2"}
    if not r2_dirs:
        return ""

    all_rows = []
    for snr in sorted(r2_dirs.keys()):
        summary = json.load(open(r2_dirs[snr] / "summary.json"))
        stability = summary.get("stability", {})
        levels = summary.get("levels", [])
        # stability keys are correlation levels > 0 (baseline is levels[0])
        stab_levels = [l for l in levels if l > 0]
        snr_label = SNR_DISPLAY.get(snr, f"SNR={snr}")

        for method in METHOD_DISPLAY:
            if method not in stability:
                continue
            row = {"SNR": snr_label, "Method": METHOD_DISPLAY[method]}
            for level in stab_levels:
                tau = stability[method].get(str(level))
                row[f"$\\rho={level}$"] = f"{tau:.3f}" if tau is not None else "---"
            all_rows.append(row)

    if not all_rows:
        return ""
    df = pd.DataFrame(all_rows)
    rho_cols = [c for c in df.columns if c.startswith("$\\rho")]
    df = _bold_best_per_column_within_group(df, "SNR", rho_cols, how="max")
    tabular = df.to_latex(index=False, escape=False)

    lines = []
    lines.append("\\begin{table}[htb]")
    lines.append("\\centering")
    lines.append("\\caption{Ranking stability (Kendall $\\tau$ between the ranking at $\\rho > 0$ and the $\\rho=0$ baseline, $R^2$ characteristic function) under varying multicollinearity and signal-to-noise ratios. Higher $\\tau$ indicates the method's ranking degrades less as inter-feature correlation increases. Most stable method per (SNR, $\\rho$) combination in \\textbf{bold}.}")
    lines.append("\\label{tab:multi_stability}")
    lines.append("\\scriptsize")
    lines.append("\\setlength{\\tabcolsep}{4pt}")
    lines.append(tabular.strip())
    lines.append("\\end{table}")
    return "\n".join(lines)


def generate_ranking_heatmaps(timestamp=None):
    """Compute Kendall-tau heatmaps between all methods (XAI + classical statistical)
    for each dataset, using R^2 rankings. Writes one EPS + one PNG per dataset to
    paper/figures/{dataset}_ranking_heatmap.pdf.

    Methods whose rankings are identical on a dataset are merged into a single
    row/column with a compound label (matching the old heatmap style).
    """
    from scipy.stats import kendalltau
    import matplotlib.pyplot as plt

    try:
        import gnuplot_style as gp
        gp.use("all")
    except ImportError:
        pass

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Short LaTeX-style labels for the heatmap (compact for axis ticks)
    xai_short = {
        "exact_cis": "Exact CIS",
        "exact_shapley": "Exact Shapley",
        "shap_explainer": "SHAP",
        "sampling_shap": "Sampling SHAP",
        "kernel_shap": "Kernel SHAP",
        "original_shapg": "Original ShapG",
        "improved_shapg": "Improved ShapG",
    }
    stat_short = {
        "beta": r"$|\beta_j|$",
        "squared_corr": r"$\rho^2_{y,x_j}$",
        "pratt": r"$\beta_j\rho_{y,x_j}$",
        "structure": r"$\rho_{x_j,\hat y}$",
        "partial": r"partial $\rho^2$",
        "semipartial": r"semipartial $\rho^2$",
        "dominance": "Dominance",
    }

    for ds in DATASETS:
        # --- XAI rankings under R^2 ---
        result_dir = find_result_dir(ds, "r2", timestamp)
        if result_dir is None:
            continue
        xai_values = json.load(open(result_dir / "feature_values.json"))

        method_rankings = []  # list of (short_label, ranking_tuple)
        for key, short in xai_short.items():
            if key not in xai_values:
                continue
            r = tuple(
                f for f, _ in sorted(xai_values[key].items(), key=lambda x: x[1], reverse=True)
            )
            method_rankings.append((short, r))

        # --- Classical statistical rankings ---
        X, y = load_dataset({"dataset": ds})
        stat_rankings = compute_all_statistical_rankings(X, y)
        for key, short in stat_short.items():
            if key not in stat_rankings:
                continue
            r = tuple(f for f, _ in stat_rankings[key])
            method_rankings.append((short, r))

        # --- Merge identical rankings ---
        merged = []  # list of (compound_label, ranking)
        seen = {}
        for label, r in method_rankings:
            if r in seen:
                seen[r].append(label)
            else:
                seen[r] = [label]
                merged.append((r, len(merged)))
        # Build compound labels in original order
        compound_labels = []
        ordered_rankings = []
        for r, _ in merged:
            labels = seen[r]
            compound_labels.append("\n".join(labels))
            ordered_rankings.append(r)

        # --- Kendall tau matrix ---
        n = len(ordered_rankings)
        tau_mat = np.zeros((n, n))
        for i in range(n):
            feats_i = list(ordered_rankings[i])
            for j in range(n):
                feats_j = list(ordered_rankings[j])
                common = [f for f in feats_i if f in feats_j]
                rank_a = [feats_i.index(f) for f in common]
                rank_b = [feats_j.index(f) for f in common]
                tau, _ = kendalltau(rank_a, rank_b)
                tau_mat[i, j] = tau if np.isfinite(tau) else 0.0

        # --- Plot ---
        fig, ax = plt.subplots(figsize=(max(6, 0.6 * n + 3), max(6, 0.6 * n + 2)))
        im = ax.imshow(tau_mat, cmap="coolwarm", vmin=-1.0, vmax=1.0, aspect="equal")
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(compound_labels, rotation=90, fontsize=8)
        ax.set_yticklabels(compound_labels, fontsize=8)
        for i in range(n):
            for j in range(n):
                val = tau_mat[i, j]
                color = "white" if abs(val) > 0.6 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        color=color, fontsize=7)
        fig.colorbar(im, ax=ax, shrink=0.8, label=r"Kendall $\tau$")
        plt.tight_layout()
        for fmt in ["pdf"]:
            plt.savefig(FIGURES_DIR / f"{ds}_ranking_heatmap.{fmt}",
                        format=fmt, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  Wrote {FIGURES_DIR}/{ds}_ranking_heatmap.pdf")


def find_near_dependency_dirs():
    """Find near_dependency_{kpi}_{timestamp} directories and group by KPI."""
    dirs = {}
    for d in RESULTS_DIR.glob("near_dependency_*"):
        if not d.is_dir():
            continue
        parts = d.name.split("_")
        if len(parts) < 4:
            continue
        # e.g., near_dependency_r2_20260418_163637 or near_dependency_f_statistic_20260418_163637
        kpi = "_".join(parts[2:-2])
        # Keep the latest timestamp per KPI
        if kpi not in dirs or d.name > dirs[kpi].name:
            dirs[kpi] = d
    return dirs


def generate_near_dependency_slopes_table():
    """One table: weighted slope S across target-max-VIF levels for all 4 KPIs.

    Rows are grouped by KPI, columns are VIF levels. Best method per (KPI, VIF)
    combination in bold.
    """
    dirs = find_near_dependency_dirs()
    if not dirs:
        return ""

    # Map file-side kpi key to display
    kpi_display = {
        "r2": "$R^2$",
        "adj_r2": "$R^2_{adj}$",
        "aic": "1/AIC",
        "f_statistic": "$F$-statistic",
    }
    kpi_order = ["r2", "adj_r2", "aic", "f_statistic"]

    # Collect rows
    method_cols = [METHOD_DISPLAY[m] for m in METHOD_DISPLAY]
    all_rows = []
    vif_targets = None
    for kpi in kpi_order:
        if kpi not in dirs:
            continue
        summary = json.load(open(dirs[kpi] / "summary.json"))
        slopes = summary.get("slopes", {})
        slopes_std = summary.get("slopes_std", {})
        if vif_targets is None:
            vif_targets = sorted(slopes.keys(), key=float)
        for method in METHOD_DISPLAY:
            if not any(method in slopes.get(t, {}) for t in vif_targets):
                continue
        # One row per method per KPI
        for method in METHOD_DISPLAY:
            row = {"KPI": kpi_display.get(kpi, kpi), "Method": METHOD_DISPLAY[method]}
            for t in vif_targets:
                s = slopes.get(t, {}).get(method)
                ss = slopes_std.get(t, {}).get(method, 0.0) if slopes_std else 0.0
                row[f"VIF={int(float(t))}"] = (
                    _format_mean_std(s, ss, kpi=kpi, std_font=_STD_FONT_SCRIPTSIZE_TABLE)
                    if s is not None else "---"
                )
            all_rows.append(row)

    if not all_rows:
        return ""
    df = pd.DataFrame(all_rows)
    vif_cols = [c for c in df.columns if c.startswith("VIF=")]
    df = _bold_best_per_column_within_group(df, "KPI", vif_cols, how="max")

    lines = []
    lines.append("\\begin{table}[htb]")
    lines.append("\\centering")
    lines.append(
        "\\caption{Weighted decreasing slope $S$ ($\\alpha = 0.8$) under the "
        "near-dependency multicollinearity regime, for all four characteristic "
        "functions and four target max-VIF levels, reported as mean $\\pm$ "
        "standard deviation over 10 seeds. Higher $S$ is better. "
        "Best method per (KPI, VIF) combination in \\textbf{bold}.}"
    )
    lines.append("\\label{tab:near_dep_slopes}")
    lines.append("\\scriptsize")
    lines.append("\\setlength{\\tabcolsep}{4pt}")
    n_vif = len(vif_cols)
    lines.append(
        f"\\begin{{tabular}}{{l{'l' * n_vif}}}"
    )
    lines.append("\\toprule")
    lines.append("Method & " + " & ".join(vif_cols) + " \\\\")

    for kpi_label in df["KPI"].drop_duplicates():
        sub = df[df["KPI"] == kpi_label]
        lines.append("\\midrule")
        lines.append(
            f"\\multicolumn{{{n_vif + 1}}}{{c}}{{\\textit{{KPI: {kpi_label}}}}} \\\\"
        )
        lines.append("\\midrule")
        for _, row in sub.iterrows():
            cells = [row["Method"]] + [row[c] for c in vif_cols]
            lines.append(" & ".join(cells) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines)


def generate_vif_table():
    """Generate LaTeX summary table of VIF across all 4 datasets.

    Uses statsmodels' variance_inflation_factor with an added intercept so that
    the values match the standard definition (and the pre-existing Abalone table).
    """
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    from statsmodels.tools.tools import add_constant

    rows = []
    for ds in DATASETS:
        X, _ = load_dataset({"dataset": ds})
        X_c = add_constant(X)
        vifs = [
            variance_inflation_factor(X_c.values, i + 1)
            for i in range(X.shape[1])
        ]
        n_high = sum(1 for v in vifs if v > 10)
        n_mid = sum(1 for v in vifs if 5 < v <= 10)
        if n_high >= max(2, X.shape[1] // 3):
            severity = "Severe"
        elif n_high > 0 or n_mid >= max(2, X.shape[1] // 3):
            severity = "Moderate"
        else:
            severity = "Low"
        rows.append({
            "Dataset": DATASET_DISPLAY.get(ds, ds),
            "n": str(X.shape[1]),
            "max": f"{max(vifs):.2f}",
            "median": f"{float(np.median(vifs)):.2f}",
            "gt10": str(n_high),
            "mid": str(n_mid),
            "severity": severity,
        })

    lines = []
    lines.append("\\begin{table}[htb]")
    lines.append("\\centering")
    lines.append(
        "\\caption{Variance inflation factors (VIF) summary for the four datasets. "
        "The conventional threshold $\\mathrm{VIF} > 10$ indicates severe multicollinearity, "
        "and $5 < \\mathrm{VIF} \\leq 10$ indicates moderate multicollinearity.}"
    )
    lines.append("\\label{tab:vif_all}")
    lines.append("\\begin{tabular}{lrrrrrl}")
    lines.append("\\toprule")
    lines.append(
        "Dataset & \\# features & max VIF & median VIF & \\# VIF $>10$ & \\# VIF $\\in (5,10]$ & Severity \\\\"
    )
    lines.append("\\midrule")
    for r in rows:
        lines.append(
            f"{r['Dataset']} & {r['n']} & {r['max']} & {r['median']} & {r['gt10']} & {r['mid']} & {r['severity']} \\\\"
        )
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines)


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Generate paper assets from results")
    parser.add_argument("--timestamp", default=None, help="Filter main-benchmark results by timestamp")
    parser.add_argument("--no-tables", action="store_true",
                        help="Skip LaTeX table generation (only copy figures + heatmaps)")
    args = parser.parse_args()

    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Copying figures ===")
    copy_figures(args.timestamp)
    copy_multicollinearity_figures()
    copy_near_dependency_figures()

    print("\n=== Generating ranking heatmaps ===")
    generate_ranking_heatmaps(args.timestamp)

    if args.no_tables:
        print("\nSkipping table generation (--no-tables).")
        print("\nDone.")
        return

    print("\n=== Generating main-benchmark tables ===")

    # All tables now generate complete \begin{table}...\end{table} environments
    tables = {
        "ranking": [(f"ranking_{ds}", generate_ranking_table(ds, args.timestamp)) for ds in DATASETS],
        "statistical": [(f"statistical_{ds}", generate_statistical_ranking_table(ds)) for ds in DATASETS],
        "timing": [("timing_all", generate_timing_table(args.timestamp))],
        "slopes": [("slopes_all", generate_slope_table(args.timestamp))],
        "ranking_stability": [("ranking_stability", generate_ranking_stability_table(args.timestamp))],
    }

    for name, content in [(n, c) for group in tables.values() for n, c in group]:
        if content:
            outfile = TABLES_DIR / f"{name}.tex"
            with open(outfile, "w") as f:
                f.write(content)
            print(f"  {outfile}")

    print("\n=== Generating multicollinearity tables ===")

    for name, func in [("multicollinearity_accuracy", generate_multicollinearity_accuracy_table),
                        ("multicollinearity_slopes", generate_multicollinearity_slopes_table),
                        ("multicollinearity_slopes_adj_r2", generate_multicollinearity_slopes_adj_r2_table),
                        ("multicollinearity_slopes_aic", generate_multicollinearity_slopes_aic_table),
                        ("multicollinearity_slopes_fstat", generate_multicollinearity_slopes_fstat_table),
                        ("multicollinearity_stability", generate_multicollinearity_stability_table),
                        ("near_dependency_slopes", generate_near_dependency_slopes_table)]:
        content = func()
        if content:
            outfile = TABLES_DIR / f"{name}.tex"
            with open(outfile, "w") as f:
                f.write(content)
            print(f"  {outfile}")

    print("\nDone.")


if __name__ == "__main__":
    main()
