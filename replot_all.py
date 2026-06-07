"""Regenerate all plots from saved experiment data without re-running anything.

Reads the latest results from ``results/`` and rewrites the plots in place,
then runs ``generate_paper_assets.py`` to copy the new PDF figures into
``paper/`` and regenerate the LaTeX tables.

Use this after updating plotting code to refresh figures without re-running
the (multi-hour) benchmark suite.

Usage::

    python replot_all.py                # replot everything
    python replot_all.py --skip-main    # skip the main feature-dropping plots
    python replot_all.py --skip-multi   # skip the block-correlation slope plots
    python replot_all.py --skip-near    # skip the near-dependency slope plots
    python replot_all.py --skip-assets  # skip the final asset-copy step
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from benchmark.plotting import DISPLAY_NAMES

try:
    import gnuplot_style as gp
    gp.use("all")
except ImportError:
    pass

ROOT = Path(__file__).parent.resolve()
RESULTS_DIR = ROOT / "results"

DATASETS = ["abalone", "admission", "concrete", "ucs_scm"]
KPIS = ["r2", "adj_r2", "aic", "f_statistic"]

# Map between filename token and code key (configs use ``fstat``, runners use ``f_statistic``)
KPI_FILENAME_ALIAS = {
    "fstat": "f_statistic",
    "f_statistic": "f_statistic",
}


def _latest_dir(pattern_re: re.Pattern[str]) -> Optional[Path]:
    candidates = [
        d for d in RESULTS_DIR.iterdir()
        if d.is_dir() and pattern_re.fullmatch(d.name)
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda d: d.name)


# --------------------------------------------------------------------------
# 1. Main benchmarks: feature_dropping plots
# --------------------------------------------------------------------------
def replot_main_benchmarks() -> int:
    """Loop over every dataset+KPI combination, calling run_benchmark.py --replot."""
    n_done = 0
    n_missing = 0
    for ds in DATASETS:
        for kpi in KPIS:
            # File config uses "fstat", but the runner saves dirs with the alias.
            # Look for either token.
            tokens = [kpi]
            if kpi == "f_statistic":
                tokens.append("fstat")
            target = None
            for tok in tokens:
                pat = re.compile(rf"{ds}_{tok}_\d{{8}}_\d{{6}}")
                target = _latest_dir(pat)
                if target:
                    break
            if not target:
                print(f"  [SKIP] no results dir for {ds}/{kpi}")
                n_missing += 1
                continue
            print(f"  [REPLOT] {target.name}")
            rc = subprocess.run(
                [sys.executable, "-u", str(ROOT / "run_benchmark.py"),
                 "--replot", str(target)],
                cwd=ROOT,
            ).returncode
            if rc != 0:
                print(f"    -> rc={rc}")
            else:
                n_done += 1
    print(f"  Main benchmarks: {n_done} replotted, {n_missing} missing")
    return n_done


# --------------------------------------------------------------------------
# 2. Block-correlation multicollinearity: slope plots
# --------------------------------------------------------------------------
def _replot_multicollinearity_dir(d: Path) -> bool:
    summary_path = d / "summary.json"
    if not summary_path.exists():
        return False
    s = json.loads(summary_path.read_text())
    levels = s.get("levels") or sorted(float(k) for k in s.get("slopes", {}).keys())
    levels_arr = np.asarray(levels, dtype=float)
    slopes_dict = s.get("slopes", {})
    slopes_std_dict = s.get("slopes_std", {}) or {}

    if not slopes_dict:
        return False
    methods = list(next(iter(slopes_dict.values())).keys())
    snr = s.get("snr", "")
    kpi = s.get("kpi", "")
    n_samples = s.get("n_samples", "")
    n_features = s.get("n_features", "")

    fig, ax = plt.subplots(figsize=(10, 6))
    for method in methods:
        slopes = [slopes_dict[str(l)][method] for l in levels]
        ax.plot(levels_arr, slopes, label=DISPLAY_NAMES.get(method, method), linewidth=1.5, alpha=0.7)
    ax.set_xlabel("Pairwise Feature Correlation ($\\rho$)", fontsize=14)
    ax.set_ylabel("Weighted Slope $S$", fontsize=14)
    ax.tick_params(labelsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=14, frameon=True, ncol=2)
    plt.tight_layout()
    for fmt in ["pdf"]:
        plt.savefig(d / f"slopes.{fmt}", format=fmt, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return True


def replot_multicollinearity() -> int:
    """Find latest multicollinearity dir per (snr, kpi) by reading summary.json
    (since recent runs no longer encode SNR in the directory name)."""
    n_done = 0
    # Match both legacy (with snr in name) and current naming.
    pat = re.compile(r"multicollinearity_(?:snr[\d.]+_)?[a-z0-9_]+_\d{8}_\d{6}")
    by_key: dict[tuple[float, str], Path] = {}
    for d in RESULTS_DIR.iterdir():
        if not (d.is_dir() and pat.fullmatch(d.name)):
            continue
        summary_path = d / "summary.json"
        if not summary_path.exists():
            continue
        try:
            s = json.loads(summary_path.read_text())
        except Exception:
            continue
        snr = float(s.get("snr", 0.0))
        kpi = str(s.get("kpi", ""))
        if not kpi:
            continue
        key = (snr, kpi)
        if key not in by_key or d.name > by_key[key].name:
            by_key[key] = d
    for (snr, kpi), d in sorted(by_key.items()):
        print(f"  [REPLOT] multicollinearity SNR={snr} KPI={kpi}: {d.name}")
        ok = _replot_multicollinearity_dir(d)
        if ok:
            n_done += 1
    print(f"  Multicollinearity: {n_done} replotted")
    return n_done


# --------------------------------------------------------------------------
# 3. Near-dependency multicollinearity: slope plots
# --------------------------------------------------------------------------
def _replot_near_dependency_dir(d: Path) -> bool:
    summary_path = d / "summary.json"
    if not summary_path.exists():
        return False
    s = json.loads(summary_path.read_text())
    targets = [float(t) for t in s.get("target_vifs", [])]
    if not targets:
        targets = sorted(float(k) for k in s.get("slopes", {}).keys())
    vif_arr = np.asarray(targets, dtype=float)
    slopes_dict = s.get("slopes", {})
    slopes_std_dict = s.get("slopes_std", {}) or {}
    if not slopes_dict:
        return False
    methods = list(next(iter(slopes_dict.values())).keys())
    kpi = s.get("kpi", "")

    fig, ax = plt.subplots(figsize=(10, 6))
    for method in methods:
        slopes = [slopes_dict[str(t)][method] for t in targets]
        ax.plot(vif_arr, slopes, marker="o", label=DISPLAY_NAMES.get(method, method), linewidth=1.5, alpha=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("Target max VIF (log scale)", fontsize=14)
    ax.set_ylabel("Weighted Slope $S$", fontsize=14)
    ax.tick_params(labelsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=14, frameon=True, ncol=2)
    plt.tight_layout()
    for fmt in ["pdf"]:
        plt.savefig(d / f"slopes.{fmt}", format=fmt, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return True


def replot_near_dependency() -> int:
    n_done = 0
    pat = re.compile(r"near_dependency_[a-z0-9_]+_\d{8}_\d{6}")
    by_kpi: dict[str, Path] = {}
    for d in RESULTS_DIR.iterdir():
        if not (d.is_dir() and pat.fullmatch(d.name)):
            continue
        m = re.match(r"near_dependency_(.+)_\d{8}_\d{6}", d.name)
        if not m:
            continue
        kpi = m.group(1)
        if kpi not in by_kpi or d.name > by_kpi[kpi].name:
            by_kpi[kpi] = d
    for kpi, d in sorted(by_kpi.items()):
        print(f"  [REPLOT] near-dependency KPI={kpi}: {d.name}")
        ok = _replot_near_dependency_dir(d)
        if ok:
            n_done += 1
    print(f"  Near-dependency: {n_done} replotted")
    return n_done


# --------------------------------------------------------------------------
def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--skip-main", action="store_true")
    p.add_argument("--skip-multi", action="store_true")
    p.add_argument("--skip-near", action="store_true")
    p.add_argument("--skip-assets", action="store_true",
                   help="Skip the final asset-copy + table-generation step entirely")
    p.add_argument("--no-tables", action="store_true",
                   help="Run asset-copy and figure-copy, but skip LaTeX table regeneration")
    args = p.parse_args()

    if not args.skip_main:
        print("\n=== [1/4] Replot main-benchmark feature-dropping plots ===")
        replot_main_benchmarks()
    if not args.skip_multi:
        print("\n=== [2/4] Replot block-correlation multicollinearity plots ===")
        replot_multicollinearity()
    if not args.skip_near:
        print("\n=== [3/4] Replot near-dependency multicollinearity plots ===")
        replot_near_dependency()
    if not args.skip_assets:
        print("\n=== [4/4] Run generate_paper_assets.py ===")
        cmd = [sys.executable, "-u", str(ROOT / "generate_paper_assets.py")]
        if args.no_tables:
            cmd.append("--no-tables")
        rc = subprocess.run(cmd, cwd=ROOT).returncode
        if rc != 0:
            print(f"  generate_paper_assets.py exited with rc={rc}")
            return rc

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
