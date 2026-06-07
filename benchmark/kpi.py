"""Feature-dropping evaluation and weighted slope computation."""

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from .characteristic import _compute_kpi


def compute_weighted_slope(metrics, beta=0.8):
    """Weighted decreasing slope with exponential decay factor beta.

    S = sum_{i=0}^{L-2} beta^i * (metrics[i] - metrics[i+1])

    A larger S means the method identifies the most important features first,
    so removing them causes the steepest initial drop in performance.
    """
    deltas = [metrics[i] - metrics[i + 1] for i in range(len(metrics) - 1)]
    weights = [beta ** i for i in range(len(deltas))]
    return np.dot(deltas, weights)


def _calculate_kpi_for_dropping(X_sub, y, kpi_type, test_size, random_state):
    """Evaluate KPI on a subset of features (retrain per drop)."""
    X_train, X_test, y_train, y_test = train_test_split(
        X_sub, y, test_size=test_size, random_state=random_state
    )
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return _compute_kpi(y_test, y_pred, kpi_type, X_train.shape[1])


def evaluate_feature_dropping(cfg, X, y, results):
    """Compute feature-dropping curves and weighted slopes for all methods.

    For each method, iterate over the per-seed rankings stored in
    ``results[name]["all_values"]`` (falling back to ``["values"]`` for
    backward compatibility), recompute the feature-dropping curve and slope
    for every seed, then aggregate as mean and std.

    Returns ``{method_name: {metrics, metrics_std, slope, slope_std,
                              slope_per_seed, n_seeds}}``.
    The ``metrics`` and ``slope`` fields hold the per-seed means so existing
    plotting and reporting code remains compatible.
    """
    eval_cfg = cfg["evaluation"]
    limit = eval_cfg["drop_limit"]
    beta = eval_cfg["weighted_slope_beta"]
    kpi_type = cfg["characteristic_function"]
    split_cfg = cfg["model"]["train_test_split"]
    test_size = split_cfg["test_size"]
    random_state = split_cfg["random_state"]

    all_features = X.columns.tolist()
    initial_kpi = _calculate_kpi_for_dropping(X, y, kpi_type, test_size, random_state)

    kpi_results = {}

    for name, r in results.items():
        all_seed_values = r.get("all_values") or [r["values"]]

        # Compute metrics once per unique ranking (deterministic methods produce
        # the same ranking across seeds; this avoids redundant retraining).
        ranking_keys = [tuple(f for f, _ in sv) for sv in all_seed_values]
        ranking_to_metrics = {}
        for ranking in ranking_keys:
            if ranking in ranking_to_metrics:
                continue
            metrics = [initial_kpi]
            for i in range(1, limit):
                features_to_drop = list(ranking[:i])
                remaining = [f for f in all_features if f not in features_to_drop]
                if remaining:
                    val = _calculate_kpi_for_dropping(
                        X[remaining], y, kpi_type, test_size, random_state
                    )
                    metrics.append(val)
                else:
                    metrics.append(0.0)
            ranking_to_metrics[ranking] = metrics

        per_seed_metrics = [ranking_to_metrics[k] for k in ranking_keys]
        per_seed_slopes = [compute_weighted_slope(m, beta) for m in per_seed_metrics]

        metrics_arr = np.array(per_seed_metrics, dtype=float)
        slopes_arr = np.array(per_seed_slopes, dtype=float)
        n_seeds = len(per_seed_slopes)

        kpi_results[name] = {
            "metrics": metrics_arr.mean(axis=0).tolist(),
            "metrics_std": (
                metrics_arr.std(axis=0, ddof=0).tolist()
                if n_seeds > 1
                else [0.0] * metrics_arr.shape[1]
            ),
            "slope": float(slopes_arr.mean()),
            "slope_std": float(slopes_arr.std(ddof=0)) if n_seeds > 1 else 0.0,
            "slope_per_seed": [float(s) for s in per_seed_slopes],
            "n_seeds": n_seeds,
        }

    return kpi_results, initial_kpi
