"""XAI method implementations."""

import random
import time
from itertools import combinations
from math import factorial

import numpy as np
import shap
from sklearn.linear_model import LinearRegression

from shapG import GraphBuilder, ShapGExplainer


def run_exact_cis(X, y, char_func, **kwargs):
    """CIS (Center of Imputation Set)."""
    all_features = X.columns.tolist()
    grand_val = char_func(all_features)
    individual_vals = {f: char_func([f]) for f in all_features}
    surplus = grand_val - sum(individual_vals.values())
    equal_share = surplus / len(all_features)
    cis_values = {f: individual_vals[f] + equal_share for f in all_features}
    return sorted(cis_values.items(), key=lambda x: x[1], reverse=True)


def run_exact_shapley(X, y, char_func, **kwargs):
    """Exact Shapley Values (combinatorial enumeration, O(2^n))."""
    all_features = X.columns.tolist()
    n = len(all_features)
    shapley_values = {}

    for feature in all_features:
        value = 0.0
        others = [f for f in all_features if f != feature]
        for i in range(n):
            for S in combinations(others, i):
                S = list(S)
                contribution = char_func(S + [feature]) - char_func(S)
                weight = factorial(len(S)) * factorial(n - len(S) - 1) / factorial(n)
                value += weight * contribution
        shapley_values[feature] = value

    return sorted(shapley_values.items(), key=lambda x: x[1], reverse=True)


def run_shap_explainer(X, y, char_func, X_train=None, X_test=None,
                       y_train=None, y_test=None, **kwargs):
    """SHAP Explainer (fast, model-aware)."""
    model = LinearRegression()
    model.fit(X_train, y_train)
    explainer = shap.Explainer(model, X_train)
    shap_values = explainer.shap_values(X_test)
    importance = np.abs(shap_values).mean(axis=0)
    features = X_train.columns.tolist()
    sorted_indices = np.argsort(importance)[::-1]
    return [(features[i], importance[i]) for i in sorted_indices]


def run_sampling_shap(X, y, char_func, X_train=None, X_test=None,
                      y_train=None, y_test=None, **kwargs):
    """Sampling SHAP."""
    model = LinearRegression()
    model.fit(X_train, y_train)
    explainer = shap.SamplingExplainer(model.predict, X_train)
    shap_values = explainer.shap_values(X_test)
    importance = np.abs(shap_values).mean(axis=0)
    features = X_train.columns.tolist()
    sorted_indices = np.argsort(importance)[::-1]
    return [(features[i], importance[i]) for i in sorted_indices]


def run_kernel_shap(X, y, char_func, X_train=None, X_test=None,
                    y_train=None, y_test=None, **kwargs):
    """Kernel SHAP."""
    model = LinearRegression()
    model.fit(X_train, y_train)
    explainer = shap.KernelExplainer(model.predict, X_train)
    shap_values = explainer.shap_values(X_test)
    importance = np.abs(shap_values).mean(axis=0)
    features = X_train.columns.tolist()
    sorted_indices = np.argsort(importance)[::-1]
    return [(features[i], importance[i]) for i in sorted_indices]


def run_original_shapg(X, y, char_func, shapg_custom_func=None, **kwargs):
    """Original ShapG (minimal spanning tree)."""
    builder = GraphBuilder()
    G = builder.from_kendalltau_minimal_edge(X, reverse=True, version='v3')
    explainer = ShapGExplainer(
        characteristic_function=shapg_custom_func,
        depth=1, n_samples=3,
        approximate_by_ratio=False, scale=False, verbose=False
    )
    sv = explainer.fit_explain(G)
    return sorted(sv.items(), key=lambda x: x[1], reverse=True)


def run_improved_shapg(X, y, char_func, shapg_custom_func=None,
                       corr_method="cosine", sim_method="cosine",
                       density_ratio=None, **kwargs):
    """Improved ShapG (rank deletion)."""
    builder = GraphBuilder()
    G = builder.from_rank_deletion(
        X, y,
        density_ratio=density_ratio,
        correlation_method=corr_method,
        similarity_method=sim_method
    )
    explainer = ShapGExplainer(
        characteristic_function=shapg_custom_func,
        depth=1, n_samples=3,
        approximate_by_ratio=False, scale=False, verbose=False
    )
    sv = explainer.fit_explain(G)
    return sorted(sv.items(), key=lambda x: x[1], reverse=True)


# Registry: name -> (function, is_stochastic)
EXPLAINER_REGISTRY = {
    "exact_cis": (run_exact_cis, False),
    "exact_shapley": (run_exact_shapley, False),
    "shap_explainer": (run_shap_explainer, False),
    "sampling_shap": (run_sampling_shap, True),
    "kernel_shap": (run_kernel_shap, True),
    "original_shapg": (run_original_shapg, False),
    "improved_shapg": (run_improved_shapg, False),
}


def run_explainers(cfg, X, y, char_func, X_train, X_test, y_train, y_test,
                   shapg_custom_func):
    """Run all enabled explainers, returning {name: {values, durations}}.

    For stochastic methods, runs multiple seeds and collects all results.
    For deterministic methods, runs once.
    """
    explainer_cfg = cfg["explainers"]
    base_seed = cfg.get("seed", 42)
    n_runs = cfg.get("n_runs", 5)
    seeds = [base_seed + i for i in range(n_runs)]
    results = {}

    for name, ecfg in explainer_cfg.items():
        if not ecfg.get("enabled", True):
            continue
        if name not in EXPLAINER_REGISTRY:
            print(f"[WARNING] Unknown explainer: {name}, skipping.")
            continue

        func, is_stochastic = EXPLAINER_REGISTRY[name]
        params = dict(ecfg.get("params", {}) or {})

        # Common kwargs passed to all explainer functions
        common_kwargs = dict(
            X_train=X_train, X_test=X_test,
            y_train=y_train, y_test=y_test,
            shapg_custom_func=shapg_custom_func,
            **params,
        )

        n_runs = len(seeds)
        all_values = []
        all_durations = []

        for run_idx in range(n_runs):
            np.random.seed(seeds[run_idx])
            random.seed(seeds[run_idx])
            label = f"  {name} (run {run_idx+1}/{n_runs})" if n_runs > 1 else f"  {name}"
            print(f"{label}...", end=" ", flush=True)

            t0 = time.time()
            values = func(X, y, char_func, **common_kwargs)
            elapsed = time.time() - t0

            all_values.append(values)
            all_durations.append(elapsed)
            print(f"{elapsed:.3f}s")

        results[name] = {
            "values": all_values[0],  # primary ranking (first run)
            "all_values": all_values,
            "durations": all_durations,
            "mean_duration": np.mean(all_durations),
            "std_duration": np.std(all_durations) if len(all_durations) > 1 else 0.0,
            "is_stochastic": is_stochastic,
            "n_runs": n_runs,
        }

    return results
