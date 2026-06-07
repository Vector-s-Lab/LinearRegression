"""Dataset loaders."""

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def _load_abalone():
    data = pd.read_csv(DATA_DIR / "abalone_processing.csv")
    X = data.drop(columns=["Rings"])
    y = data["Rings"]
    return X, y


def _load_admission():
    data = pd.read_csv(DATA_DIR / "admission.csv")
    X = data.drop(columns=["Chance of Admit "])
    y = data["Chance of Admit "]
    return X, y


def _load_concrete():
    data = pd.read_csv(DATA_DIR / "concrete.csv")
    X = data.drop(columns=["strength"])
    y = data["strength"]
    return X, y


def _load_ucs_scm():
    data = pd.read_csv(DATA_DIR / "ucs_scm.csv")
    X = data.drop(columns=["strength"])
    y = data["strength"]
    return X, y


DATASETS = {
    "abalone": _load_abalone,
    "admission": _load_admission,
    "concrete": _load_concrete,
    "ucs_scm": _load_ucs_scm,
}


def load_dataset(cfg):
    """Load dataset specified in config."""
    name = cfg["dataset"]
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset '{name}'. Available: {list(DATASETS.keys())}")
    return DATASETS[name]()
