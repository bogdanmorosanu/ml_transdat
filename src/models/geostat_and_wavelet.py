"""Zero-PyTorch geostatistical, patch-RBF, and wavelet shift regressors.

Implements:
- ``LocalKrigingShiftModel`` (V2-11): Local anisotropic Matern 3/2 Ordinary Kriging
  with calibrated predictive uncertainty ``predict_with_std(coords)``.
- ``PartitionOfUnityRBFShiftModel`` (V2-12a): Overlapping Wendland C2-weighted
  Partition of Unity RBF patches.
- ``WaveletGridShiftModel`` (V2-12b): Pure-NumPy 2D Haar Discrete Wavelet Transform
  sparse residual compressor and bilinear reconstructor.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.interpolate import RBFInterpolator
from scipy.spatial import cKDTree

from src.models.base import BaseGridShiftModel


class LocalKrigingShiftModel(BaseGridShiftModel):
    """Local anisotropic Matern 3/2 Ordinary Kriging with uncertainty estimation."""

    def __init__(
        self,
        neighbors: int = 25,
        lengthscale_n: float = 2000.0,
        lengthscale_e: float = 2000.0,
        nugget: float = 1e-6,
        kernel: str = "matern32",
    ) -> None:
        super().__init__()
        k_norm = kernel.lower()
        if k_norm not in ("matern32", "matern52"):
            raise ValueError(f"Unsupported kernel='{kernel}'. Expected 'matern32' or 'matern52'.")
        self.neighbors = int(neighbors)
        self.lengthscale_n = float(lengthscale_n)
        self.lengthscale_e = float(lengthscale_e)
        self.nugget = float(nugget)
        self.kernel = k_norm

        self._coords: np.ndarray | None = None
        self._shifts: np.ndarray | None = None
        self._kdtree: cKDTree | None = None
        self._prior_var: np.ndarray | None = None

    def _kernel_eval(self, r: np.ndarray) -> np.ndarray:
        if self.kernel == "matern32":
            sr = math.sqrt(3.0) * r
            return (1.0 + sr) * np.exp(-sr)
        sr = math.sqrt(5.0) * r
        return (1.0 + sr + (5.0 / 3.0) * (r * r)) * np.exp(-sr)

    def _fit_impl(self, coords: np.ndarray, shifts: np.ndarray) -> None:
        self._coords = np.asarray(coords, dtype=np.float64)
        self._shifts = np.asarray(shifts, dtype=np.float64)
        scaled = np.column_stack(
            [self._coords[:, 0] / self.lengthscale_n, self._coords[:, 1] / self.lengthscale_e]
        )
        self._kdtree = cKDTree(scaled)
        self._prior_var = np.maximum(np.var(self._shifts, axis=0), 1e-8)

    def predict_with_std(self, coords: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Predict 2D shifts and per-axis Kriging standard deviations ``(M, 2)``."""
        if self._coords is None or self._shifts is None or self._kdtree is None or self._prior_var is None:
            raise RuntimeError("Model is not fitted yet.")
        q_all = np.asarray(coords, dtype=np.float64)
        n_q = len(q_all)
        k = min(self.neighbors, len(self._coords))
        preds = np.empty((n_q, 2), dtype=np.float64)
        stds = np.empty((n_q, 2), dtype=np.float64)

        q_scaled_all = np.column_stack(
            [q_all[:, 0] / self.lengthscale_n, q_all[:, 1] / self.lengthscale_e]
        )
        eye_k = np.eye(k, dtype=np.float64) * max(self.nugget, 1e-9)

        chunk = 1024
        for start in range(0, n_q, chunk):
            end = min(start + chunk, n_q)
            qs = q_scaled_all[start:end]  # (B, 2)
            dists_q, idxs = self._kdtree.query(qs, k=k)
            if k == 1:
                dists_q = dists_q[:, None]
                idxs = idxs[:, None]

            pts_loc = np.stack(
                [self._coords[idxs, 0] / self.lengthscale_n, self._coords[idxs, 1] / self.lengthscale_e],
                axis=-1,
            )  # (B, k, 2)
            y_loc = self._shifts[idxs]  # (B, k, 2)

            diff = pts_loc[:, :, None, :] - pts_loc[:, None, :, :]
            r_mat = np.sqrt(np.sum(diff * diff, axis=-1))  # (B, k, k)
            k_mat = self._kernel_eval(r_mat) + eye_k[None, :, :]
            k_star = self._kernel_eval(dists_q)  # (B, k)

            # Ordinary Kriging: K * w + mu * 1 = k_star, 1^T w = 1
            b_sz = end - start
            aug = np.zeros((b_sz, k + 1, k + 1), dtype=np.float64)
            aug[:, :k, :k] = k_mat
            aug[:, :k, k] = 1.0
            aug[:, k, :k] = 1.0
            rhs = np.ones((b_sz, k + 1), dtype=np.float64)
            rhs[:, :k] = k_star

            sol = np.linalg.solve(aug, rhs[:, :, None])[..., 0]  # (B, k + 1)
            w = sol[:, :k]  # (B, k)
            mu = sol[:, k]  # (B,)
            preds[start:end] = np.einsum("bk,bkd->bd", w, y_loc)

            # Normalized Kriging variance: 1 - sum(w * k_star) - mu
            var_norm = np.maximum(1.0 - np.sum(w * k_star, axis=1) - mu, 0.0)
            stds[start:end] = np.sqrt(var_norm[:, None] * self._prior_var[None, :])

        return preds, stds

    def _predict_impl(self, coords: np.ndarray) -> np.ndarray:
        preds, _ = self.predict_with_std(coords)
        return preds

    def _runtime_dependencies(self) -> tuple[str, ...]:
        return ("numpy", "scipy")


class PartitionOfUnityRBFShiftModel(BaseGridShiftModel):
    """Overlapping Wendland C2-weighted Partition of Unity RBF regressor."""

    def __init__(
        self,
        patch_grid: tuple[int, int] = (10, 10),
        overlap: float = 1.6,
        kernel: str = "cubic",
        smoothing: float = 1e-6,
    ) -> None:
        super().__init__()
        self.patch_grid = (int(patch_grid[0]), int(patch_grid[1]))
        self.overlap = float(overlap)
        self.kernel = str(kernel)
        self.smoothing = float(smoothing)

        self._patches: list[tuple[np.ndarray, np.ndarray, RBFInterpolator]] = []
        self._fallback_mean: np.ndarray | None = None

    def _wendland_c2(self, r: np.ndarray) -> np.ndarray:
        t = np.clip(1.0 - r, 0.0, 1.0)
        return (t**4) * (4.0 * r + 1.0)

    def _fit_impl(self, coords: np.ndarray, shifts: np.ndarray) -> None:
        x_in = np.asarray(coords, dtype=np.float64)
        y_in = np.asarray(shifts, dtype=np.float64)
        self._fallback_mean = np.mean(y_in, axis=0)
        self._patches = []

        min_xy = np.min(x_in, axis=0)
        max_xy = np.max(x_in, axis=0)
        span = np.maximum(max_xy - min_xy, 1.0)
        nx, ny = self.patch_grid
        rad_n = 0.5 * span[0] / max(nx - 1, 1) * self.overlap if nx > 1 else span[0]
        rad_e = 0.5 * span[1] / max(ny - 1, 1) * self.overlap if ny > 1 else span[1]
        radii = np.array([rad_n, rad_e], dtype=np.float64)

        cn = np.linspace(min_xy[0], max_xy[0], nx)
        ce = np.linspace(min_xy[1], max_xy[1], ny)
        degree = 1 if self.kernel in ("cubic", "thin_plate_spline") else 0

        for n0 in cn:
            for e0 in ce:
                center = np.array([n0, e0], dtype=np.float64)
                rel = (x_in - center) / radii
                r = np.sqrt(np.sum(rel * rel, axis=1))
                mask = r <= 1.0
                if int(np.sum(mask)) < 10:
                    continue
                interp = RBFInterpolator(
                    rel[mask],
                    y_in[mask],
                    kernel=self.kernel,
                    degree=degree,
                    smoothing=self.smoothing,
                )
                self._patches.append((center, radii, interp))

    def _predict_impl(self, coords: np.ndarray) -> np.ndarray:
        if self._fallback_mean is None:
            raise RuntimeError("Model is not fitted yet.")
        q_all = np.asarray(coords, dtype=np.float64)
        n_q = len(q_all)
        num = np.zeros((n_q, 2), dtype=np.float64)
        den = np.zeros(n_q, dtype=np.float64)

        for center, radii, interp in self._patches:
            rel = (q_all - center) / radii
            r = np.sqrt(np.sum(rel * rel, axis=1))
            mask = r < 1.0
            if not np.any(mask):
                continue
            w = self._wendland_c2(r[mask])
            num[mask] += w[:, None] * interp(rel[mask])
            den[mask] += w

        out = np.empty((n_q, 2), dtype=np.float64)
        valid = den > 0.0
        out[valid] = num[valid] / den[valid, None]
        if not np.all(valid):
            out[~valid] = self._fallback_mean
        return out

    def _runtime_dependencies(self) -> tuple[str, ...]:
        return ("numpy", "scipy")


class WaveletGridShiftModel(BaseGridShiftModel):
    """Pure-NumPy 2D Haar DWT sparse residual compressor + bilinear interpolator."""

    def __init__(
        self,
        grid_step_m: float = 687.5,
        levels: int = 3,
        threshold_mm: float = 0.2,
    ) -> None:
        super().__init__()
        self.grid_step_m = float(grid_step_m)
        self.levels = int(levels)
        self.threshold_mm = float(threshold_mm)

        self._min_n: float = 0.0
        self._min_e: float = 0.0
        self._orig_h: int = 0
        self._orig_w: int = 0
        self._pad_h: int = 0
        self._pad_w: int = 0
        self._sparse_idx: tuple[np.ndarray, np.ndarray] | None = None
        self._sparse_vals: tuple[np.ndarray, np.ndarray] | None = None
        self._recon_grid: np.ndarray | None = None

    def _dwt2_haar(self, img: np.ndarray) -> np.ndarray:
        out = img.copy()
        h, w = out.shape
        for _ in range(self.levels):
            if h < 2 or w < 2:
                break
            sub = out[:h, :w]
            ll = 0.5 * (sub[0::2, 0::2] + sub[0::2, 1::2] + sub[1::2, 0::2] + sub[1::2, 1::2])
            lh = 0.5 * (sub[0::2, 0::2] - sub[0::2, 1::2] + sub[1::2, 0::2] - sub[1::2, 1::2])
            hl = 0.5 * (sub[0::2, 0::2] + sub[0::2, 1::2] - sub[1::2, 0::2] - sub[1::2, 1::2])
            hh = 0.5 * (sub[0::2, 0::2] - sub[0::2, 1::2] - sub[1::2, 0::2] + sub[1::2, 1::2])
            hh2, hw2 = h // 2, w // 2
            out[:hh2, :hw2] = ll
            out[:hh2, hw2:w] = lh
            out[hh2:h, :hw2] = hl
            out[hh2:h, hw2:w] = hh
            h, w = hh2, hw2
        return out

    def _idwt2_haar(self, coeff: np.ndarray) -> np.ndarray:
        out = coeff.copy()
        h_sizes = []
        w_sizes = []
        h, w = out.shape
        for _ in range(self.levels):
            if h < 2 or w < 2:
                break
            h_sizes.append(h)
            w_sizes.append(w)
            h //= 2
            w //= 2

        for h_cur, w_cur in reversed(list(zip(h_sizes, w_sizes))):
            hh2, hw2 = h_cur // 2, w_cur // 2
            ll = out[:hh2, :hw2]
            lh = out[:hh2, hw2:w_cur]
            hl = out[hh2:h_cur, :hw2]
            hh = out[hh2:h_cur, hw2:w_cur]
            rec = np.empty((h_cur, w_cur), dtype=np.float64)
            rec[0::2, 0::2] = 0.5 * (ll + lh + hl + hh)
            rec[0::2, 1::2] = 0.5 * (ll - lh + hl - hh)
            rec[1::2, 0::2] = 0.5 * (ll + lh - hl - hh)
            rec[1::2, 1::2] = 0.5 * (ll - lh - hl + hh)
            out[:h_cur, :w_cur] = rec
        return out

    def _ensure_recon_grid(self) -> np.ndarray:
        if self._recon_grid is not None:
            return self._recon_grid
        assert self._sparse_idx is not None and self._sparse_vals is not None
        recon = np.empty((self._orig_h, self._orig_w, 2), dtype=np.float64)
        for ch in range(2):
            c_mat = np.zeros(self._pad_h * self._pad_w, dtype=np.float64)
            c_mat[self._sparse_idx[ch]] = self._sparse_vals[ch].astype(np.float64)
            grid_ch = self._idwt2_haar(c_mat.reshape(self._pad_h, self._pad_w))
            recon[:, :, ch] = grid_ch[: self._orig_h, : self._orig_w]
        self._recon_grid = recon
        return recon

    def _fit_impl(self, coords: np.ndarray, shifts: np.ndarray) -> None:
        x_in = np.asarray(coords, dtype=np.float64)
        y_in = np.asarray(shifts, dtype=np.float64)
        self._min_n = float(np.min(x_in[:, 0]))
        self._min_e = float(np.min(x_in[:, 1]))
        max_n = float(np.max(x_in[:, 0]))
        max_e = float(np.max(x_in[:, 1]))

        self._orig_h = max(round((max_n - self._min_n) / self.grid_step_m) + 1, 2)
        self._orig_w = max(round((max_e - self._min_e) / self.grid_step_m) + 1, 2)
        mult = 2 ** self.levels
        self._pad_h = int(math.ceil(self._orig_h / mult) * mult)
        self._pad_w = int(math.ceil(self._orig_w / mult) * mult)

        grid = np.zeros((self._pad_h, self._pad_w, 2), dtype=np.float64)
        counts = np.zeros((self._pad_h, self._pad_w), dtype=np.float64)
        ri = np.clip(np.round((x_in[:, 0] - self._min_n) / self.grid_step_m).astype(np.int64), 0, self._orig_h - 1)
        ci = np.clip(np.round((x_in[:, 1] - self._min_e) / self.grid_step_m).astype(np.int64), 0, self._orig_w - 1)
        np.add.at(grid[:, :, 0], (ri, ci), y_in[:, 0])
        np.add.at(grid[:, :, 1], (ri, ci), y_in[:, 1])
        np.add.at(counts, (ri, ci), 1.0)
        nz = counts > 0
        grid[nz] /= counts[nz, None]
        if self._orig_h < self._pad_h:
            grid[self._orig_h :, :, :] = grid[self._orig_h - 1 : self._orig_h, :, :]
        if self._orig_w < self._pad_w:
            grid[:, self._orig_w :, :] = grid[:, self._orig_w - 1 : self._orig_w, :]

        thresh = self.threshold_mm * 1e-3
        idxs: list[np.ndarray] = []
        vals: list[np.ndarray] = []
        for ch in range(2):
            c_mat = self._dwt2_haar(grid[:, :, ch]).ravel()
            keep = np.flatnonzero(np.abs(c_mat) >= thresh)
            idxs.append(keep.astype(np.uint32))
            vals.append(c_mat[keep].astype(np.float32))

        self._sparse_idx = (idxs[0], idxs[1])
        self._sparse_vals = (vals[0], vals[1])
        self._recon_grid = None

    def _predict_impl(self, coords: np.ndarray) -> np.ndarray:
        recon = self._ensure_recon_grid()
        q_all = np.asarray(coords, dtype=np.float64)
        fn = np.clip((q_all[:, 0] - self._min_n) / self.grid_step_m, 0.0, self._orig_h - 1.000001)
        fe = np.clip((q_all[:, 1] - self._min_e) / self.grid_step_m, 0.0, self._orig_w - 1.000001)
        r0 = np.floor(fn).astype(np.int64)
        c0 = np.floor(fe).astype(np.int64)
        u = (fn - r0)[:, None]
        v = (fe - c0)[:, None]
        return (
            (1.0 - u) * (1.0 - v) * recon[r0, c0]
            + (1.0 - u) * v * recon[r0, c0 + 1]
            + u * (1.0 - v) * recon[r0 + 1, c0]
            + u * v * recon[r0 + 1, c0 + 1]
        )

    def _runtime_dependencies(self) -> tuple[str, ...]:
        return ("numpy",)

    def _grid_lookup_bytes(self) -> int:
        if self._sparse_idx is None or self._sparse_vals is None:
            return 0
        return int(
            sum(a.nbytes for a in self._sparse_idx)
            + sum(a.nbytes for a in self._sparse_vals)
        )
