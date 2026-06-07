"""Model factory."""

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


MODEL_CLASSES = {
    "LinearRegression": LinearRegression,
}


def build_model(cfg):
    """Instantiate a model from config."""
    mcfg = cfg["model"]
    model_type = mcfg["type"]
    if model_type not in MODEL_CLASSES:
        raise ValueError(f"Unknown model '{model_type}'. Available: {list(MODEL_CLASSES.keys())}")
    params = mcfg.get("params", {}) or {}
    return MODEL_CLASSES[model_type](**params)
