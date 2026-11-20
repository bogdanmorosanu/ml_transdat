"""Multi-tier evaluation of adaptive Quadtree grids on the corridor benchmark."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pytransdat.trans_ro import TransRO


def build_indexer(grid: pd.DataFrame):
    return (
        grid["n_min"].values,
        grid["n_max"].values,
        grid["e_min"].values,
        grid["e_max"].values,
        grid["v00"].values,
        grid["v01"].values,
        grid["v10"].values,
        grid["v11"].values,
    )


def query_grid(indexer, n_vals: np.ndarray, e_vals: np.ndarray) -> np.ndarray:
    n_min, n_max, e_min, e_max, v00, v01, v10, v11 = indexer
    preds = np.full(len(n_vals), np.nan, dtype=float)
    for i, (n, e) in enumerate(zip(n_vals, e_vals)):
        m = (n_min <= n) & (n <= n_max) & (e_min <= e) & (e <= e_max)
        if not m.any():
            continue
        idx = int(np.argmax(m))
        u = (n - n_min[idx]) / (n_max[idx] - n_min[idx])
        v = (e - e_min[idx]) / (e_max[idx] - e_min[idx])
        preds[i] = (
            (1 - u) * (1 - v) * v00[idx]
            + (1 - u) * v * v01[idx]
            + u * (1 - v) * v10[idx]
            + u * v * v11[idx]
        )
    return preds


def summarize(cm: np.ndarray) -> dict[str, float]:
    return {
        "mean_cm": float(np.mean(cm)),
        "median_cm": float(np.median(cm)),
        "p95_cm": float(np.percentile(cm, 95)),
        "p99_cm": float(np.percentile(cm, 99)),
        "max_cm": float(np.max(cm)),
    }


def main() -> None:
    exp_dir = Path("experiments/adaptive_quadtree_grid")
    plots_dir = exp_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    tr = TransRO()
    df_cor = pd.read_parquet("data/benchmarks/ro_corridor_snw.parquet")
    n_pts = df_cor["st70_n"].values
    e_pts = df_cor["st70_e"].values

    corr_n, corr_e = tr._t_gr2d.interp(n_pts.tolist(), e_pts.tolist())
    true_dn = np.array(corr_n)
    true_de = np.array(corr_e)

    tiers = ["2cm", "1cm", "3mm", "1mm"]
    all_results = {}
    err_2d_by_tier = {}

    for tier in tiers:
        dn_grid = pd.read_parquet(f"data/grids/adaptive/adaptive_{tier}_dn.parquet")
        de_grid = pd.read_parquet(f"data/grids/adaptive/adaptive_{tier}_de.parquet")

        dn_idx = build_indexer(dn_grid)
        de_idx = build_indexer(de_grid)

        p_dn = query_grid(dn_idx, n_pts, e_pts)
        p_de = query_grid(de_idx, n_pts, e_pts)

        val = ~(np.isnan(p_dn) | np.isnan(p_de))
        edn = np.abs(true_dn[val] - p_dn[val]) * 100.0
        ede = np.abs(true_de[val] - p_de[val]) * 100.0
        e2d = np.hypot(edn, ede)

        err_2d_by_tier[tier] = e2d
        all_results[tier] = {
            "dn_leaves": int(len(dn_grid)),
            "de_leaves": int(len(de_grid)),
            "matched": int(val.sum()),
            "total": int(len(n_pts)),
            "dn": summarize(edn),
            "de": summarize(ede),
            "2d": summarize(e2d),
        }

    with open(exp_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    # Multi-tier comparison bar plot
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(tiers))
    width = 0.25

    means = [all_results[t]["2d"]["mean_cm"] for t in tiers]
    p95s = [all_results[t]["2d"]["p95_cm"] for t in tiers]
    maxs = [all_results[t]["2d"]["max_cm"] for t in tiers]

    ax.bar(x - width, means, width, label="Mean 2D Error", color="#2563eb")
    ax.bar(x, p95s, width, label="P95 2D Error", color="#d97706")
    ax.bar(x + width, maxs, width, label="Max 2D Error", color="#dc2626")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{t}\n({all_results[t]['dn_leaves']:,} cells)" for t in tiers])
    ax.set_ylabel("Error (cm)")
    ax.set_yscale("log")
    ax.set_title("Adaptive Quadtree Accuracy Across Tolerances (Corridor Benchmark)")
    ax.legend()
    ax.grid(alpha=0.3, which="both", axis="y")
    fig.tight_layout()
    fig.savefig(plots_dir / "tier_comparison.png", dpi=180)
    plt.close(fig)

    print(json.dumps(all_results, indent=2))


if __name__ == "__main__":
    main()
