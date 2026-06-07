"""
Find the best density_ratio for Improved ShapG via greedy edge addition.

Strategy:
1. Build complete weighted graph and compute the edge deletion order
   (same as from_rank_deletion internally)
2. Start from the sparsest connected graph (maximum deletion)
3. Add edges back one batch at a time (reverse deletion order)
4. After each batch: run ShapG → compute weighted slope (R²)
5. Stop when slope has no positive improvement for 2 consecutive batches

The characteristic function is cached across all ShapG runs, so later
runs reuse coalition evaluations from earlier ones.

Usage:
    python find_best_ratio.py --dataset abalone
    python find_best_ratio.py --dataset admission
    python find_best_ratio.py --dataset concrete
"""

import argparse
import json
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from shapG import GraphBuilder, ShapGExplainer, CustomFunction

from benchmark.data import load_dataset
from benchmark.kpi import compute_weighted_slope, _calculate_kpi_for_dropping

import matplotlib.pyplot as plt

try:
    import gnuplot_style as gp
    gp.use("all")
except ImportError:
    pass


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


def build_cached_char_func(X, y, kpi_type="r2", test_size=0.2, random_state=42):
    """Build a characteristic function with coalition-level caching.

    The cache persists across multiple ShapG runs, so coalitions evaluated
    in earlier runs are reused in later ones.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    _cache = {}
    _stats = {"hits": 0, "misses": 0}

    def char_func_raw(features):
        cols = list(features) if features else []
        if len(cols) == 0:
            return 0.0
        model = LinearRegression()
        model.fit(X_train[cols], y_train)
        y_pred = model.predict(X_test[cols])
        return _compute_kpi(y_test, y_pred, kpi_type, len(cols))

    def wrapper(coalition, context):
        key = frozenset(coalition)
        if key in _cache:
            _stats["hits"] += 1
            return _cache[key]
        _stats["misses"] += 1
        val = char_func_raw(list(coalition))
        _cache[key] = val
        return val

    kpi_names = {"r2": "Regression R2", "aic": "Regression 1/AIC", "f_statistic": "Regression F-statistic"}
    custom_func = CustomFunction(wrapper, name=f"{kpi_names.get(kpi_type, kpi_type)} (cached)")

    return custom_func, _cache, _stats


def get_deletion_order(X, y, corr_method="cosine", sim_method="cosine"):
    """Reproduce the edge deletion order from from_rank_deletion.

    Returns:
        complete_graph: the full weighted graph
        deleted_edges: list of (u, v, weight) in deletion order
                       (first deleted = least important edge)
    """
    features = X.columns.tolist()
    n_features = len(features)

    # Get feature ranking (same as from_rank_deletion)
    feature_rank_indices = GraphBuilder.get_feature_rank(X, y, corr_method)
    feature_rank = [features[i] for i in feature_rank_indices]

    # Calculate similarity matrix
    X_array = X.to_numpy()
    similarity_matrix = GraphBuilder._calculate_similarity_matrix(X_array, sim_method)

    # Build complete graph
    G = nx.Graph()
    for i in range(n_features):
        for j in range(i + 1, n_features):
            weight = similarity_matrix[i, j]
            G.add_edge(features[i], features[j], weight=weight)

    complete_graph = G.copy()

    # Simulate deletion (same logic as from_rank_deletion, but record order)
    deleted_edges = []
    for node in feature_rank:
        node_edges = [(u, v) for u, v in G.edges(node)]
        node_edges.sort(key=lambda x: G[x[0]][x[1]]["weight"], reverse=True)

        for edge in node_edges:
            weight = G[edge[0]][edge[1]]["weight"]
            G.remove_edge(*edge)
            if nx.is_connected(G):
                deleted_edges.append((edge[0], edge[1], weight))
            else:
                G.add_edge(*edge, weight=weight)

    # G is now the sparsest connected graph
    return complete_graph, G, deleted_edges


def evaluate_slope(X, y, graph, shapg_custom_func, kpi_type="r2",
                   drop_limit=7, beta=0.8, test_size=0.2, random_state=42):
    """Run ShapG on a graph and compute the weighted slope.

    Uses seed=42 to match the benchmark evaluation exactly.
    """
    import random as _random

    np.random.seed(random_state)
    _random.seed(random_state)

    explainer = ShapGExplainer(
        characteristic_function=shapg_custom_func,
        depth=1, n_samples=3,
        approximate_by_ratio=False, scale=False, verbose=False
    )
    sv = explainer.fit_explain(graph)
    feature_rank = [f for f, _ in sorted(sv.items(), key=lambda x: x[1], reverse=True)]

    all_features = X.columns.tolist()
    initial_kpi = _calculate_kpi_for_dropping(X, y, kpi_type, test_size, random_state)
    metrics = [initial_kpi]

    for i in range(1, drop_limit):
        features_to_drop = feature_rank[:i]
        remaining = [f for f in all_features if f not in features_to_drop]
        if remaining:
            metrics.append(_calculate_kpi_for_dropping(X[remaining], y, kpi_type, test_size, random_state))
        else:
            metrics.append(0.0)

    slope = compute_weighted_slope(metrics, beta)
    return slope, feature_rank, metrics


def main():
    parser = argparse.ArgumentParser(description="Find best density_ratio for Improved ShapG")
    parser.add_argument("--dataset", default="abalone", choices=["abalone", "admission", "concrete", "ucs_scm"])
    parser.add_argument("--kpi", default="r2", choices=["r2", "adj_r2", "aic", "f_statistic"])
    parser.add_argument("--patience", type=int, default=2, help="Stop after N batches without improvement")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("results") / f"ratio_search_{args.dataset}_{args.kpi}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = {"dataset": args.dataset}
    X, y = load_dataset(cfg)
    n_features = X.shape[1]
    max_edges = n_features * (n_features - 1) // 2
    drop_limit = n_features  # drop up to n-1 features (keep 1), matches benchmark

    # Max ratio: for small datasets (n<=10) search the full space.
    # For larger datasets, cap at k * min_ratio to keep graphs meaningfully sparse.
    K_MULTIPLIER = 3
    MAX_RATIO_CAP = 0.5
    min_ratio = (n_features - 1) / max_edges if max_edges > 0 else 1.0
    if n_features <= 10:
        max_ratio = 1.0
    else:
        max_ratio = min(K_MULTIPLIER * min_ratio, MAX_RATIO_CAP)
    max_search_edges = int(max_ratio * max_edges)

    print(f"Dataset: {args.dataset} ({X.shape[0]} samples, {n_features} features)")
    print(f"Max edges: {max_edges}, Min edges (spanning tree): {n_features - 1}")
    print(f"Search max ratio: {max_ratio:.3f} ({max_search_edges} edges) — {K_MULTIPLIER}x min_ratio, cap={MAX_RATIO_CAP}")

    print(f"KPI: {args.kpi}")

    # Build cached characteristic function
    shapg_custom_func, cache, stats = build_cached_char_func(X, y, kpi_type=args.kpi)

    # Get deletion order
    print("\nComputing edge deletion order...")
    complete_graph, sparse_graph, deleted_edges = get_deletion_order(X, y)

    # Trim edges_to_add to max_search_edges
    n_addable = max_search_edges - sparse_graph.number_of_edges()
    n_addable = max(0, min(n_addable, len(deleted_edges)))

    print(f"  Sparsest graph: {sparse_graph.number_of_edges()} edges")
    print(f"  Edges to search: {n_addable} (of {len(deleted_edges)} total)")

    # Batch size: aim for ~10 evaluation steps
    batch_size = max(1, n_addable // 10)
    print(f"  Batch size: {batch_size}")

    # --- Greedy edge addition ---
    print("\n" + "=" * 60)
    print("GREEDY EDGE ADDITION SEARCH")
    print("=" * 60)

    current_graph = sparse_graph.copy()
    # Reverse deletion order: last deleted = should be added first
    # Only consider edges up to max_search_edges
    edges_to_add = list(reversed(deleted_edges))[:n_addable]

    results = []
    best_slope = -np.inf
    best_n_edges = None
    best_ratio = None
    best_ranking = None
    no_improvement_count = 0

    # Evaluate starting point (sparsest graph)
    print(f"\n[Step 0] edges={current_graph.number_of_edges()}, ratio={current_graph.number_of_edges()/max_edges:.3f}")
    t0 = time.time()
    slope, ranking, metrics = evaluate_slope(
        X, y, current_graph, shapg_custom_func, args.kpi, drop_limit
    )
    elapsed = time.time() - t0
    ratio = current_graph.number_of_edges() / max_edges
    print(f"  slope={slope:.6f}  time={elapsed:.2f}s  cache: {stats['hits']} hits, {stats['misses']} misses")

    results.append({
        "step": 0, "n_edges": current_graph.number_of_edges(),
        "ratio": ratio, "slope": slope, "ranking": ranking,
        "time": elapsed,
    })
    best_slope = slope
    best_n_edges = current_graph.number_of_edges()
    best_ratio = ratio
    best_ranking = ranking

    step = 1
    edge_idx = 0
    while edge_idx < len(edges_to_add):
        # Add a batch of edges
        batch_end = min(edge_idx + batch_size, len(edges_to_add))
        for i in range(edge_idx, batch_end):
            u, v, w = edges_to_add[i]
            current_graph.add_edge(u, v, weight=w)
        edge_idx = batch_end

        ratio = current_graph.number_of_edges() / max_edges
        print(f"\n[Step {step}] edges={current_graph.number_of_edges()}, ratio={ratio:.3f}")

        t0 = time.time()
        slope, ranking, metrics = evaluate_slope(
            X, y, current_graph, shapg_custom_func, args.kpi, drop_limit
        )
        elapsed = time.time() - t0
        print(f"  slope={slope:.6f}  time={elapsed:.2f}s  cache: {stats['hits']} hits, {stats['misses']} misses")

        results.append({
            "step": step, "n_edges": current_graph.number_of_edges(),
            "ratio": ratio, "slope": slope, "ranking": ranking,
            "time": elapsed,
        })

        if slope > best_slope:
            improvement = slope - best_slope
            print(f"  ** NEW BEST (improvement: +{improvement:.6f})")
            best_slope = slope
            best_n_edges = current_graph.number_of_edges()
            best_ratio = ratio
            best_ranking = ranking
            no_improvement_count = 0
        else:
            no_improvement_count += 1
            print(f"  No improvement ({no_improvement_count}/{args.patience})")
            if no_improvement_count >= args.patience:
                print(f"\n  Early stopping: no improvement for {args.patience} consecutive batches")
                break

        step += 1

    # --- Compute a safe ratio that survives 4-decimal YAML truncation ---
    # from_rank_deletion uses: target_edges = int(density_ratio * max_edges)
    # If target_edges < min_edges, it auto-adjusts to 1.5x minimum.
    # We need: int(round4(ratio) * max_edges) >= best_n_edges
    # Safe formula: (best_n_edges + 0.5) / max_edges, then round up to 4 decimals
    import math
    safe_ratio = math.ceil((best_n_edges + 0.5) / max_edges * 10000) / 10000
    # Verify
    G_verify = GraphBuilder.from_rank_deletion(
        X, y, density_ratio=safe_ratio,
        correlation_method="cosine", similarity_method="cosine"
    )
    verified_ratio = safe_ratio
    if G_verify.number_of_edges() != best_n_edges:
        print(f"\n[WARNING] Safe ratio {safe_ratio:.4f} produces {G_verify.number_of_edges()} edges "
              f"(expected {best_n_edges}). Searching...")
        # Brute-force search over 4-decimal ratios
        for r_int in range(int(best_ratio * 10000), 10001):
            candidate = r_int / 10000
            G_test = GraphBuilder.from_rank_deletion(
                X, y, density_ratio=candidate,
                correlation_method="cosine", similarity_method="cosine"
            )
            if G_test.number_of_edges() == best_n_edges:
                verified_ratio = candidate
                break
        print(f"  Found ratio: {verified_ratio:.4f} → {best_n_edges} edges")

    # --- Results ---
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"  Best edges:    {best_n_edges}/{max_edges}")
    print(f"  Best ratio:    {verified_ratio:.4f} (verified via from_rank_deletion)")
    print(f"  Best slope:    {best_slope:.6f}")
    print(f"  Best ranking:  {best_ranking}")
    print(f"  Cache stats:   {stats['hits']} hits, {stats['misses']} misses "
          f"({stats['hits']/(stats['hits']+stats['misses'])*100:.1f}% hit rate)")

    # --- Plot slope vs ratio ---
    fig, ax = plt.subplots(figsize=(10, 6))
    ratios = [r["ratio"] for r in results]
    slopes = [r["slope"] for r in results]
    ax.plot(ratios, slopes, "o-", linewidth=1.5, alpha=0.7)
    ax.axvline(x=best_ratio, color="r", linestyle="--", alpha=0.5, label=f"Best ratio={verified_ratio:.4f} ({best_n_edges} edges)")
    ax.set_xlabel("Density Ratio", fontsize=12)
    kpi_labels = {"r2": "$R^2$", "aic": "1/AIC", "f_statistic": "F-statistic"}
    ax.set_ylabel(f"Weighted Slope $S$ ({kpi_labels.get(args.kpi, args.kpi)}, $\\beta$=0.8)", fontsize=12)
    ax.set_title(f"Improved ShapG: Density Ratio Search — {args.dataset} ({args.kpi})")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    for fmt in ["eps", "png"]:
        plt.savefig(output_dir / f"ratio_search.{fmt}", format=fmt, dpi=300, bbox_inches="tight")
    plt.close()

    # --- Save summary ---
    summary = {
        "dataset": args.dataset,
        "kpi": args.kpi,
        "n_features": n_features,
        "max_edges": max_edges,
        "best_ratio": verified_ratio,
        "search_ratio": best_ratio,
        "best_n_edges": best_n_edges,
        "best_slope": best_slope,
        "best_ranking": best_ranking,
        "cache_hits": stats["hits"],
        "cache_misses": stats["misses"],
        "steps": [{k: v for k, v in r.items()} for r in results],
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to {output_dir}/")
    print(f"\nTo use this ratio, set density_ratio: {verified_ratio:.4f} in your config YAML.")


if __name__ == "__main__":
    main()
