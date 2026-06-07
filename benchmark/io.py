"""Config loading and result saving."""

import json
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml


def load_config(config_path):
    """Load YAML config file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def save_results(output_dir, explainer_results, kpi_results, initial_kpi, cfg):
    """Save all results to output directory.

    Files written:
        config.yaml          - copy of the config used
        timing.json           - {method: {mean, std, n_runs, durations}}
        ranking.csv           - feature ranking comparison table
        kpi.json              - {method: {metrics, slope}}
        feature_values.json   - {method: {feature: value}}
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(output_dir / "config.yaml", "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    # Timing
    timing = {}
    for name, r in explainer_results.items():
        timing[name] = {
            "mean": r["mean_duration"],
            "std": r["std_duration"],
            "n_runs": r["n_runs"],
            "durations": r["durations"],
        }
    with open(output_dir / "timing.json", "w") as f:
        json.dump(timing, f, indent=2)

    # Feature ranking table
    methods = {}
    for name, r in explainer_results.items():
        methods[name] = [feat for feat, _ in r["values"]]
    df = pd.DataFrame(methods)
    df.insert(0, "Rank", [f"Top{i+1}" for i in range(len(df))])

    # Add timing row
    time_row = {"Rank": "Time(s)"}
    for name, r in explainer_results.items():
        mean_t = r["mean_duration"]
        std_t = r["std_duration"]
        if r["n_runs"] > 1:
            time_row[name] = f"{mean_t:.3f} +/- {std_t:.3f}"
        else:
            time_row[name] = f"{mean_t:.3f}"
    df = pd.concat([df, pd.DataFrame([time_row])], ignore_index=True)
    df.to_csv(output_dir / "ranking.csv", index=False)

    # KPI results
    kpi_data = {"initial_kpi": initial_kpi}
    for name, kpi in kpi_results.items():
        kpi_data[name] = {
            "metrics": kpi["metrics"],
            "metrics_std": kpi.get("metrics_std", [0.0] * len(kpi["metrics"])),
            "slope": kpi["slope"],
            "slope_std": kpi.get("slope_std", 0.0),
            "slope_per_seed": kpi.get("slope_per_seed", [kpi["slope"]]),
            "n_seeds": kpi.get("n_seeds", 1),
        }
    with open(output_dir / "kpi.json", "w") as f:
        json.dump(kpi_data, f, indent=2)

    # Feature importance values
    values_data = {}
    for name, r in explainer_results.items():
        values_data[name] = {feat: float(val) for feat, val in r["values"]}
    with open(output_dir / "feature_values.json", "w") as f:
        json.dump(values_data, f, indent=2)

    print(f"\nResults saved to {output_dir}/")


def load_results(result_dir):
    """Load previously saved results for replotting.

    Returns (cfg, explainer_results, kpi_results, initial_kpi).
    """
    result_dir = Path(result_dir)

    cfg = load_config(result_dir / "config.yaml")

    with open(result_dir / "timing.json", "r") as f:
        timing = json.load(f)

    with open(result_dir / "kpi.json", "r") as f:
        kpi_data = json.load(f)

    with open(result_dir / "feature_values.json", "r") as f:
        values_data = json.load(f)

    initial_kpi = kpi_data.pop("initial_kpi")

    # Reconstruct explainer_results from timing + feature_values
    explainer_results = {}
    for name, t in timing.items():
        values_dict = values_data.get(name, {})
        sorted_values = sorted(values_dict.items(), key=lambda x: x[1], reverse=True)
        explainer_results[name] = {
            "values": sorted_values,
            "durations": t["durations"],
            "mean_duration": t["mean"],
            "std_duration": t["std"],
            "n_runs": t["n_runs"],
        }

    return cfg, explainer_results, kpi_data, initial_kpi
