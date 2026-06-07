"""Classical statistical methods for feature importance in linear regression.

Implements seven classical predictor-importance measures:
    1. Standardized regression coefficient |beta_j|
    2. Squared product-moment correlation rho^2_{y,x_j}
    3. Pratt's measure: beta_j * rho_{y,x_j} (with standardized beta)
    4. Structure coefficient: rho_{x_j, y_hat} / rho_{y, y_hat}
    5. Squared partial correlation
    6. Squared semipartial correlation (= drop in R^2 when x_j is removed)
    7. Dominance analysis (= exact Shapley on R^2)

All functions take a pandas DataFrame X, a pandas Series y, and return a
list of (feature, value) sorted by descending importance.
"""

from itertools import combinations
from math import factorial

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split


def _fit_on_train(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test


def standardized_beta(X, y):
    """Absolute value of standardized regression coefficient."""
    X_std = (X - X.mean()) / X.std()
    y_std = (y - y.mean()) / y.std()
    model = LinearRegression().fit(X_std, y_std)
    coefs = dict(zip(X.columns, np.abs(model.coef_)))
    return sorted(coefs.items(), key=lambda kv: kv[1], reverse=True)


def squared_correlation(X, y):
    """Squared Pearson correlation rho^2_{y,x_j}."""
    vals = {col: np.corrcoef(y, X[col])[0, 1] ** 2 for col in X.columns}
    return sorted(vals.items(), key=lambda kv: kv[1], reverse=True)


def pratt(X, y):
    """Pratt's measure: standardized_beta_j * rho_{y,x_j}."""
    X_std = (X - X.mean()) / X.std()
    y_std = (y - y.mean()) / y.std()
    model = LinearRegression().fit(X_std, y_std)
    corrs = {col: np.corrcoef(y, X[col])[0, 1] for col in X.columns}
    vals = {col: abs(model.coef_[i] * corrs[col]) for i, col in enumerate(X.columns)}
    return sorted(vals.items(), key=lambda kv: kv[1], reverse=True)


def structure_coefficient(X, y):
    """rho_{x_j, y_hat} — correlation between feature and predicted y."""
    model = LinearRegression().fit(X, y)
    y_hat = model.predict(X)
    vals = {col: abs(np.corrcoef(X[col], y_hat)[0, 1]) for col in X.columns}
    return sorted(vals.items(), key=lambda kv: kv[1], reverse=True)


def squared_partial_correlation(X, y):
    """Squared partial correlation of each feature with y, controlling for others.

    For feature j: r^2_{yj.rest} = t_j^2 / (t_j^2 + df),
    where t_j is the t-statistic of coefficient j and df = n - p - 1.
    """
    from sklearn.linear_model import LinearRegression as LR
    n, p = X.shape
    X_np = X.to_numpy(dtype=float)
    y_np = y.to_numpy(dtype=float)
    X_aug = np.column_stack([np.ones(n), X_np])
    beta, *_ = np.linalg.lstsq(X_aug, y_np, rcond=None)
    residuals = y_np - X_aug @ beta
    df = n - p - 1
    mse = np.sum(residuals ** 2) / df if df > 0 else np.nan
    cov_beta = mse * np.linalg.inv(X_aug.T @ X_aug)
    se = np.sqrt(np.diag(cov_beta))
    t_stats = beta / se
    # Skip intercept at index 0
    vals = {}
    for i, col in enumerate(X.columns, start=1):
        t = t_stats[i]
        r2_partial = (t ** 2) / (t ** 2 + df) if df > 0 else 0.0
        vals[col] = r2_partial
    return sorted(vals.items(), key=lambda kv: kv[1], reverse=True)


def squared_semipartial_correlation(X, y):
    """Squared semipartial correlation: decrease in model R^2 when x_j is removed.

    sr^2_j = R^2(full) - R^2(without x_j).
    """
    model_full = LinearRegression().fit(X, y)
    r2_full = r2_score(y, model_full.predict(X))
    vals = {}
    for col in X.columns:
        X_drop = X.drop(columns=[col])
        model = LinearRegression().fit(X_drop, y)
        r2 = r2_score(y, model.predict(X_drop))
        vals[col] = max(0.0, r2_full - r2)
    return sorted(vals.items(), key=lambda kv: kv[1], reverse=True)


def dominance_analysis(X, y, test_size=0.2, random_state=42):
    """Dominance analysis based on Shapley value over R^2 coalitions.

    Equivalent to Exact Shapley with R^2 as the characteristic function.
    We evaluate R^2 on a held-out test set (to match the characteristic function
    used by the XAI pipeline in benchmark/characteristic.py, so that Dominance
    and Exact Shapley produce identical rankings).

    Each coalition value is computed at most once via a frozenset cache.
    For n features, there are 2^n unique coalitions to evaluate.
    """
    features = list(X.columns)
    n = len(features)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    cache = {frozenset(): 0.0}

    def v(S):
        key = frozenset(S)
        if key not in cache:
            cols = list(S)
            model = LinearRegression().fit(X_train[cols], y_train)
            cache[key] = r2_score(y_test, model.predict(X_test[cols]))
        return cache[key]

    vals = {f: 0.0 for f in features}
    # Precompute marginal weights to avoid repeated factorial calls
    weights = {k: factorial(k) * factorial(n - k - 1) / factorial(n) for k in range(n)}
    for f in features:
        others = [x for x in features if x != f]
        for k in range(n):
            w = weights[k]
            for S in combinations(others, k):
                vals[f] += w * (v(list(S) + [f]) - v(list(S)))
    return sorted(vals.items(), key=lambda kv: kv[1], reverse=True)


STATISTICAL_METHODS = {
    "beta": ("$|\\beta_j|$", standardized_beta),
    "squared_corr": ("$\\rho^2_{y,x_j}$", squared_correlation),
    "pratt": ("$\\beta_j \\rho_{y,x_j}$", pratt),
    "structure": ("$\\rho_{x_j,\\hat{y}}$", structure_coefficient),
    "partial": ("partial $\\rho^2$", squared_partial_correlation),
    "semipartial": ("semipartial $\\rho^2$", squared_semipartial_correlation),
    "dominance": ("Dominance", dominance_analysis),
}


def compute_all_statistical_rankings(X, y):
    """Return dict {method_key: [(feature, value), ...]} for all 7 methods."""
    return {key: fn(X, y) for key, (_, fn) in STATISTICAL_METHODS.items()}
