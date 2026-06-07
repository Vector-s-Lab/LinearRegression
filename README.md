# Feature Importance Methods for Linear Regression

Benchmark comparing Shapley-based and CIS-based feature-importance methods on
linear regression, across four real datasets and two controlled
multicollinearity regimes.

Methods evaluated:

- **Exact CIS** — Center of Imputation Set, an O(n) allocation rule from
  cooperative game theory.
- **Exact Shapley** — full 2^n coalition enumeration.
- **SHAP family** — SHAP Explainer, Sampling SHAP, Kernel SHAP (via `shap`).
- **ShapG family** — Original and Improved ShapG (via `shapG`).

Each method is scored under four characteristic functions (`R²`, `R²_adj`,
`1/AIC`, `F-statistic`) by a weighted decreasing slope `S` (α = 0.8) on a
feature-dropping curve.

## Installation

```bash
git clone <repo-url>
cd LinearRegression
pip install -r requirements.txt
```

Requires Python ≥ 3.9. `shapG` ships on TestPyPI; the `--extra-index-url`
directive in `requirements.txt` handles this automatically.

## Datasets

Four tabular regression datasets are bundled under `data/`. See
[`data/README.md`](data/README.md) for source links and licences.

## Usage

```bash
# Single config (16 bundled: 4 datasets × 4 KPIs)
python run_benchmark.py configs/abalone_r2.yaml

# Full suite (16 main + 12 block-correlation + 4 near-dependency)
bash run_all.sh

# Individual synthetic studies
python run_multicollinearity.py --snr 2.38 --kpi r2
python run_near_dependency.py --kpi r2 --target-vifs 10 50 100 500

# Replot figures from saved results, skipping table regeneration
python replot_all.py --no-tables
```

Results land in `results/<dataset>_<kpi>_<timestamp>/`. `run_all.sh` accepts
`--skip-main`, `--skip-multi`, `--skip-near`, `--skip-assets` to run only
part of the suite.

## Repository layout

```
benchmark/                  Core library (CIS, Shapley, SHAP, ShapG, KPIs, plotting)
configs/                    16 YAML configs (4 datasets × 4 KPIs)
data/                       4 tabular datasets (CSV)
run_benchmark.py            Main entry point for one config
run_multicollinearity.py    Block-correlation synthetic study
run_near_dependency.py      Near-dependency synthetic study
run_all.sh                  One-click driver for the full suite
generate_paper_assets.py    LaTeX table and figure generation
replot_all.py               Refresh figures without re-running experiments
```

## Notes

- **SHAP Explainer** uses the closed-form linear-model branch of the `shap`
  library: `φ_i(x) = β_i (x_i − x̄_i)`. After mean-absolute aggregation the
  importance reduces to `|β_i| · MAD(x_i)`, closely related to the
  standardised regression coefficient.
- **Sampling/Kernel SHAP** use the SHAP library's default `nsamples="auto"`
  and on linear models converge to the SHAP Explainer closed form on every
  dataset/KPI.
- **ShapG** hyperparameters are set per-method in the YAML configs.
- Each experiment runs 10 seeds (`42, …, 51`); deterministic methods give
  std = 0, the graph-based methods expose genuine seed-to-seed variability.

## License

MIT — see [`LICENSE`](LICENSE).
