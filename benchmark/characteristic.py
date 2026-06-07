"""Characteristic functions (R2, Adjusted R2, AIC, F-statistic)."""

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from shapG import CustomFunction


def _compute_kpi(y_test, y_pred, kpi_type, n_features):
    """Compute a single KPI value."""
    if kpi_type == "r2":
        return r2_score(y_test, y_pred)

    elif kpi_type == "adj_r2":
        r2 = r2_score(y_test, y_pred)
        n = len(y_test)
        p = n_features
        if p == 0 or n <= p + 1:
            return r2
        return 1 - (1 - r2) * (n - 1) / (n - p - 1)

    elif kpi_type == "aic":
        n = len(y_test)
        rss = np.sum((y_test - y_pred) ** 2)
        mse = rss / n
        L = -(n / 2) * np.log(2 * np.pi) - (n / 2) * np.log(mse) - 0.5
        k = n_features + 1
        aic = 2 * k - 2 * L
        return 1 / aic if aic != 0 else 0

    elif kpi_type == "f_statistic":
        n = len(y_test)
        p = n_features
        y_mean = np.mean(y_test)
        tss = np.sum((y_test - y_mean) ** 2)
        rss = np.sum((y_test - y_pred) ** 2)
        if rss == 0 or (n - p - 1) <= 0:
            return 0.0
        return ((tss - rss) / p) / (rss / (n - p - 1))

    else:
        raise ValueError(f"Unknown kpi_type: {kpi_type}")


def build_char_func(cfg, X, y):
    """Build a characteristic function closure for Shapley/CIS computation.

    Returns (char_func, X_train, X_test, y_train, y_test).
    char_func(features) -> float.
    """
    kpi_type = cfg["characteristic_function"]
    split_cfg = cfg["model"]["train_test_split"]
    test_size = split_cfg["test_size"]
    random_state = split_cfg["random_state"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    def char_func(features):
        cols = list(features) if features else []
        if len(cols) == 0:
            return 0.0
        model = LinearRegression()
        model.fit(X_train[cols], y_train)
        y_pred = model.predict(X_test[cols])
        return _compute_kpi(y_test, y_pred, kpi_type, len(cols))

    return char_func, X_train, X_test, y_train, y_test


def build_shapg_custom_function(cfg, X, y):
    """Build a shapG CustomFunction adapter."""
    kpi_type = cfg["characteristic_function"]
    split_cfg = cfg["model"]["train_test_split"]
    test_size = split_cfg["test_size"]
    random_state = split_cfg["random_state"]
    n_total = X.shape[1]

    def wrapper(coalition, context):
        cols = list(coalition)
        if len(cols) == 0:
            return 0.0
        X_train, X_test, y_train, y_test = train_test_split(
            X[cols], y, test_size=test_size, random_state=random_state
        )
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        return _compute_kpi(y_test, y_pred, kpi_type, n_total)

    kpi_names = {"r2": "Regression R2", "aic": "Regression 1/AIC", "f_statistic": "Regression F-statistic"}
    return CustomFunction(wrapper, name=kpi_names.get(kpi_type, kpi_type))
