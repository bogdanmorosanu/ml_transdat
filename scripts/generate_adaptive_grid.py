"""Generate per-component adaptive Quadtree grids (dN, dE) in Stereo70 (N, E).

Directly models the official Romanian .spg 2D grid shifts via pytransdat.
Pure one-way formulation: (N, E) -> (corr_n, corr_e).
No PyProj, no projection roundtrips, no two-way checks.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from pytransdat.exceptions import NoDataGridErr, OutOfGrid2DErr
from pytransdat.trans_ro import TransRO

# Romania .spg 2D grid bounds in Stereo70 (meters)
N_MIN, N_MAX = 213634.564, 785634.564
E_MIN, E_MAX = 109783.04, 890783.04


def make_shift_fn():
    """Return a cached function (n, e) -> (corr_n, corr_e) directly from pytransdat."""
    tr = TransRO()
    t_gr2d = tr._t_gr2d
    cache: dict[tuple[float, float], tuple[float, float] | None] = {}

    def get_shift(n: float, e: float) -> tuple[float, float] | None:
        key = (round(n, 3), round(e, 3))
        if key in cache:
            return cache[key]
        try:
            corr_n, corr_e = t_gr2d.interp([n], [e])
            val = (corr_n[0], corr_e[0])
        except (OutOfGrid2DErr, NoDataGridErr):
            val = None
        cache[key] = val
        return val

    return get_shift


def build_component_quadtree(
    get_shift,
    comp_idx: int,
    tol: float,
    min_depth: int = 4,
    max_depth: int = 15,
) -> pd.DataFrame:
    """Build a Quadtree for a single shift component (0=corr_n, 1=corr_e)."""
    stack = [(N_MIN, N_MAX, E_MIN, E_MAX, 0)]
    leaves: list[dict[str, float | int]] = []

    while stack:
        n0, n1, e0, e1, d = stack.pop()
        nm, em = 0.5 * (n0 + n1), 0.5 * (e0 + e1)

        c00 = get_shift(n0, e0)
        c01 = get_shift(n0, e1)
        c10 = get_shift(n1, e0)
        c11 = get_shift(n1, e1)
        cc = get_shift(nm, em)

        valid = [c for c in (c00, c01, c10, c11, cc) if c is not None]
        if not valid:
            if d < min_depth:
                stack.extend(
                    [
                        (n0, nm, e0, em, d + 1),
                        (n0, nm, em, e1, d + 1),
                        (nm, n1, e0, em, d + 1),
                        (nm, n1, em, e1, d + 1),
                    ]
                )
            continue

        all_valid = len(valid) == 5
        if not all_valid:
            if d < max_depth:
                stack.extend(
                    [
                        (n0, nm, e0, em, d + 1),
                        (n0, nm, em, e1, d + 1),
                        (nm, n1, e0, em, d + 1),
                        (nm, n1, em, e1, d + 1),
                    ]
                )
            continue

        v00 = c00[comp_idx]
        v01 = c01[comp_idx]
        v10 = c10[comp_idx]
        v11 = c11[comp_idx]
        vc = cc[comp_idx]

        # Primary check: center vs bilinear average of 4 corners
        interp = 0.25 * (v00 + v01 + v10 + v11)
        err = abs(vc - interp)

        # Secondary check: 4 sub-quadrant centers (u, v in {0.25, 0.75})
        if err <= tol and d >= min_depth:
            for u, n_sub in (
                (0.25, n0 + 0.25 * (n1 - n0)),
                (0.75, n0 + 0.75 * (n1 - n0)),
            ):
                for v, e_sub in (
                    (0.25, e0 + 0.25 * (e1 - e0)),
                    (0.75, e0 + 0.75 * (e1 - e0)),
                ):
                    cs = get_shift(n_sub, e_sub)
                    if cs is None:
                        err = float("inf")
                        break
                    b_interp = (
                        (1 - u) * (1 - v) * v00
                        + (1 - u) * v * v01
                        + u * (1 - v) * v10
                        + u * v * v11
                    )
                    err = max(err, abs(cs[comp_idx] - b_interp))
                if err > tol:
                    break

        if (d < min_depth or err > tol) and d < max_depth:
            stack.extend(
                [
                    (n0, nm, e0, em, d + 1),
                    (n0, nm, em, e1, d + 1),
                    (nm, n1, e0, em, d + 1),
                    (nm, n1, em, e1, d + 1),
                ]
            )
        else:
            leaves.append(
                {
                    "n_min": n0,
                    "n_max": n1,
                    "e_min": e0,
                    "e_max": e1,
                    "v00": v00,
                    "v01": v01,
                    "v10": v10,
                    "v11": v11,
                    "depth": d,
                    "center_err": err,
                }
            )

    return pd.DataFrame(leaves)


def generate_for_tolerance(
    get_shift, tol: float, tag: str, out_dir: Path, max_depth: int = 15
) -> None:
    """Generate and save both dN and dE grids for a given tolerance."""
    print(
        f"\n--- Generating Adaptive Grids for {tag} (tol={tol:.4f} m, max_depth={max_depth}) ---"
    )
    for name, idx in [("dn", 0), ("de", 1)]:
        t0 = time.perf_counter()
        df = build_component_quadtree(
            get_shift, comp_idx=idx, tol=tol, max_depth=max_depth
        )
        dt = time.perf_counter() - t0
        out_path = out_dir / f"adaptive_{tag}_{name}.parquet"
        df.to_parquet(out_path, index=False)
        sizes_n = df["n_max"] - df["n_min"]
        sizes_e = df["e_max"] - df["e_min"]
        print(
            f"[{name.upper()}] leaves={len(df):,}, "
            f"depth={df['depth'].min()}..{df['depth'].max()}, "
            f"cell_N={sizes_n.min():.1f}..{sizes_n.max():.1f}m, "
            f"cell_E={sizes_e.min():.1f}..{sizes_e.max():.1f}m, "
            f"time={dt:.2f}s -> {out_path}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Adaptive Quadtree grid generator.")
    parser.add_argument(
        "--tolerances",
        nargs="+",
        default=["2cm", "1cm", "3mm", "1mm"],
        choices=["2cm", "1cm", "3mm", "1mm"],
        help="Tolerance presets to generate.",
    )
    args = parser.parse_args()

    tol_map = {
        "2cm": (0.02, 12),
        "1cm": (0.01, 13),
        "3mm": (0.003, 14),
        "1mm": (0.001, 15),
    }

    out_dir = Path("data/grids/adaptive")
    out_dir.mkdir(parents=True, exist_ok=True)

    get_shift = make_shift_fn()
    for tag in args.tolerances:
        tol_val, max_d = tol_map[tag]
        generate_for_tolerance(
            get_shift, tol=tol_val, tag=tag, out_dir=out_dir, max_depth=max_d
        )


if __name__ == "__main__":
    main()
