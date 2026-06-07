"""Benchmark modules for XAI methods comparison."""

from .data import load_dataset
from .models import build_model
from .characteristic import build_char_func, build_shapg_custom_function
from .explainers import run_explainers, EXPLAINER_REGISTRY
from .kpi import compute_weighted_slope, evaluate_feature_dropping
from .io import load_config, load_results, save_results
from .plotting import plot_feature_dropping, plot_ranking_table
