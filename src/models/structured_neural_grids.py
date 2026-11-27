"""Structured neural-geometric and factorized-grid shift regressors.

Implements:
- ``NeuralBicubicHeadShiftModel`` (V2-5): Predicts 16 bicubic monomial coefficients
  per axis inside each 11 km SPG cell via either a compact coefficient grid or a
  zero-lookup cell-center ResNet.
- ``KPlanesGridShiftModel`` (V2-7): Low-rank 1D Northing x Easting factorized plane
  representation decoded by a compact SiLU MLP.
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from src.geometry.cell_uv import encode_cyclic_cell_uv, get_spg_grid_bounds
from src.models._features import SpatialFeatureEncoder
from src.models._torch_trainer import train_torch_regressor
from src.models.base import BaseGridShiftModel
from src.models.torch_mlp import TorchMLPNet


def _bicubic_monomials(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Evaluate 16-term bicubic monomial basis ``u^i * v^j`` for ``i,j in 0..3``."""
    u2 = u * u
    u3 = u2 * u
    v2 = v * v
    v3 = v2 * v
    us = (np.ones_like(u), u, u2, u3)
    vs = (np.ones_like(v), v, v2, v3)
    cols = [ui * vj for ui in us for vj in vs]
    return np.column_stack(cols).astype(np.float32)


class _BicubicGridHeadNet(nn.Module):
    def __init__(self, n_rows: int, n_cols: int) -> None:
        super().__init__()
        self.coeffs = nn.Parameter(torch.zeros(n_rows, n_cols, 2, 16))
        nn.init.normal_(self.coeffs, mean=0.0, std=1e-4)

    def forward(self, cell_rc: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
        r = cell_rc[:, 0]
        c = cell_rc[:, 1]
        c_sel = self.coeffs[r, c]  # (B, 2, 16)
        return torch.einsum("bdk,bk->bd", c_sel, phi)


class _BicubicMLPHeadNet(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_res_blocks: int,
    ) -> None:
        super().__init__()
        self.backbone = TorchMLPNet(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            num_res_blocks=num_res_blocks,
            out_dim=32,
            activation="silu",
        )

    def forward(self, cell_feats: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
        coeffs = self.backbone(cell_feats).view(-1, 2, 16)
        return torch.einsum("bdk,bk->bd", coeffs, phi)


class NeuralBicubicHeadShiftModel(BaseGridShiftModel):
    """Neural predictor outputting 16 bicubic ``(u, v)`` monomial coefficients per cell."""

    def __init__(
        self,
        backend: str = "grid",
        hidden_dim: int = 128,
        num_res_blocks: int = 2,
        fourier_scales: tuple[float, ...] = (1.0, 3.0, 10.0),
        num_fourier_frequencies: int = 48,
        batch_size: int = 1024,
        lr: float = 3e-3,
        weight_decay: float = 1e-6,
        max_epochs: int = 150,
        patience: int = 25,
        val_fraction: float = 0.0,
        loss_type: str = "huber",
        device: str | None = None,
        random_state: int = 42,
        spg_path: str | None = None,
    ) -> None:
        super().__init__()
        b_norm = backend.lower()
        if b_norm not in ("grid", "mlp"):
            raise ValueError(f"Unsupported backend='{backend}'. Expected 'grid' or 'mlp'.")
        self.backend = b_norm
        self.hidden_dim = int(hidden_dim)
        self.num_res_blocks = int(num_res_blocks)
        self.fourier_scales = tuple(float(s) for s in fourier_scales)
        self.num_fourier_frequencies = int(num_fourier_frequencies)
        self.batch_size = int(batch_size)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.max_epochs = int(max_epochs)
        self.patience = int(patience)
        self.val_fraction = float(val_fraction)
        self.loss_type = str(loss_type).lower()
        self.random_state = int(random_state)
        self.spg_path = spg_path
        self.device = (
            ("cuda" if torch.cuda.is_available() else "cpu") if device is None else str(device)
        )

        self._min_n: float = 0.0
        self._min_e: float = 0.0
        self._step_n: float = 11000.0
        self._step_e: float = 11000.0
        self._n_cell_rows: int = 52
        self._n_cell_cols: int = 71
        self.encoder: SpatialFeatureEncoder | None = None
        self.net: nn.Module | None = None

    def _init_bounds(self) -> None:
        min_e, max_e, min_n, max_n, step_e, step_n = get_spg_grid_bounds(self.spg_path)
        self._min_n = min_n
        self._min_e = min_e
        self._step_n = step_n
        self._step_e = step_e
        self._n_cell_rows = max(round((max_n - min_n) / step_n), 1)
        self._n_cell_cols = max(round((max_e - min_e) / step_e), 1)

    def _cell_coords_and_phi(
        self, coords: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n_arr = np.asarray(coords[:, 0], dtype=np.float64)
        e_arr = np.asarray(coords[:, 1], dtype=np.float64)
        r_idx = np.clip(
            np.floor((n_arr - self._min_n) / self._step_n).astype(np.int64),
            0,
            self._n_cell_rows - 1,
        )
        c_idx = np.clip(
            np.floor((e_arr - self._min_e) / self._step_e).astype(np.int64),
            0,
            self._n_cell_cols - 1,
        )
        u = (n_arr - (self._min_n + r_idx * self._step_n)) / self._step_n
        v = (e_arr - (self._min_e + c_idx * self._step_e)) / self._step_e
        phi = _bicubic_monomials(u, v)
        cell_rc = np.column_stack([r_idx, c_idx]).astype(np.int64)
        cell_centers = np.column_stack(
            [
                self._min_n + (r_idx + 0.5) * self._step_n,
                self._min_e + (c_idx + 0.5) * self._step_e,
            ]
        )
        return cell_rc, cell_centers, phi

    def _fit_impl(self, coords: np.ndarray, shifts: np.ndarray) -> None:
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        self._init_bounds()
        cell_rc, cell_centers, phi = self._cell_coords_and_phi(coords)
        targets = shifts.astype(np.float32)

        if self.backend == "grid":
            self.net = _BicubicGridHeadNet(self._n_cell_rows, self._n_cell_cols).to(self.device)
            inputs: tuple[np.ndarray, ...] = (cell_rc, phi)
        else:
            self.encoder = SpatialFeatureEncoder(
                normalize_inputs=True,
                use_fourier_features=True,
                fourier_scales=self.fourier_scales,
                num_fourier_frequencies=self.num_fourier_frequencies,
                use_cyclic_cell_pos=False,
                random_state=self.random_state,
            )
            cell_feats = self.encoder.fit_transform(cell_centers).astype(np.float32)
            self.net = _BicubicMLPHeadNet(
                in_dim=cell_feats.shape[1],
                hidden_dim=self.hidden_dim,
                num_res_blocks=self.num_res_blocks,
            ).to(self.device)
            inputs = (cell_feats, phi)

        train_torch_regressor(
            net=self.net,
            forward_fn=self.net,
            inputs=inputs,
            targets=targets,
            batch_size=self.batch_size,
            lr=self.lr,
            weight_decay=self.weight_decay,
            max_epochs=self.max_epochs,
            patience=self.patience,
            val_fraction=self.val_fraction,
            loss_type=self.loss_type,
            device=self.device,
            random_state=self.random_state,
        )

    def _predict_impl(self, coords: np.ndarray) -> np.ndarray:
        if self.net is None:
            raise RuntimeError("Model is not fitted yet.")
        cell_rc, cell_centers, phi = self._cell_coords_and_phi(coords)
        if self.backend == "grid":
            first_arr = cell_rc
        else:
            if self.encoder is None:
                raise RuntimeError("Encoder is not initialized.")
            first_arr = self.encoder.transform(cell_centers).astype(np.float32)

        self.net.eval()
        preds: list[np.ndarray] = []
        chunk_size = 16384
        with torch.no_grad():
            for i in range(0, len(phi), chunk_size):
                b0 = torch.from_numpy(first_arr[i : i + chunk_size]).to(self.device)
                b1 = torch.from_numpy(phi[i : i + chunk_size]).to(self.device)
                preds.append(self.net(b0, b1).cpu().numpy())
        return np.vstack(preds).astype(np.float64)

    def _runtime_dependencies(self) -> tuple[str, ...]:
        return ("torch", "numpy")

    def _grid_lookup_bytes(self) -> int:
        if self.backend != "grid" or self.net is None:
            return 0
        assert isinstance(self.net, _BicubicGridHeadNet)
        return int(self.net.coeffs.numel() * self.net.coeffs.element_size())

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        if self.net is not None:
            state["net"] = copy.deepcopy(self.net).cpu()
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            self.device = "cpu"
        if self.net is not None:
            self.net = self.net.to(self.device)
            self.net.eval()


# ---------------------------------------------------------------------------
# V2-7: K-Planes Low-Rank Factorized 1D Axis Planes
# ---------------------------------------------------------------------------


class _KPlanesNet(nn.Module):
    def __init__(self, n_bins: int, feature_dim: int, aux_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.plane_n = nn.Parameter(torch.randn(1, feature_dim, 1, n_bins) * 1e-2)
        self.plane_e = nn.Parameter(torch.randn(1, feature_dim, 1, n_bins) * 1e-2)
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim + aux_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, norm_ne: torch.Tensor, aux: torch.Tensor) -> torch.Tensor:
        # grid_sample 4D expects (N, C, H=1, W) and grid (N, H_out=1, W_out=B, 2)
        b_sz = norm_ne.shape[0]
        zeros = torch.zeros(b_sz, device=norm_ne.device, dtype=norm_ne.dtype)
        grid_n = torch.stack([norm_ne[:, 0], zeros], dim=-1).view(1, 1, b_sz, 2)
        grid_e = torch.stack([norm_ne[:, 1], zeros], dim=-1).view(1, 1, b_sz, 2)
        fn = F.grid_sample(self.plane_n, grid_n, mode="bilinear", padding_mode="border", align_corners=True)
        fe = F.grid_sample(self.plane_e, grid_e, mode="bilinear", padding_mode="border", align_corners=True)
        fn_flat = fn.squeeze(0).squeeze(1).transpose(0, 1)
        fe_flat = fe.squeeze(0).squeeze(1).transpose(0, 1)
        return self.mlp(torch.cat([fn_flat * fe_flat, aux], dim=-1))


class KPlanesGridShiftModel(BaseGridShiftModel):
    """Low-rank 1D Northing x Easting factorized plane (K-Planes) shift regressor."""

    def __init__(
        self,
        n_bins: int = 128,
        feature_dim: int = 16,
        hidden_dim: int = 64,
        cyclic_cell_harmonics: int = 2,
        batch_size: int = 1024,
        lr: float = 3e-3,
        weight_decay: float = 1e-6,
        max_epochs: int = 150,
        patience: int = 25,
        val_fraction: float = 0.0,
        loss_type: str = "huber",
        device: str | None = None,
        random_state: int = 42,
        spg_path: str | None = None,
    ) -> None:
        super().__init__()
        self.n_bins = int(n_bins)
        self.feature_dim = int(feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.cyclic_cell_harmonics = int(cyclic_cell_harmonics)
        self.batch_size = int(batch_size)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.max_epochs = int(max_epochs)
        self.patience = int(patience)
        self.val_fraction = float(val_fraction)
        self.loss_type = str(loss_type).lower()
        self.random_state = int(random_state)
        self.spg_path = spg_path
        self.device = (
            ("cuda" if torch.cuda.is_available() else "cpu") if device is None else str(device)
        )

        self._min_n: float = 0.0
        self._max_n: float = 1.0
        self._min_e: float = 0.0
        self._max_e: float = 1.0
        self.net: _KPlanesNet | None = None

    def _encode_inputs(self, coords: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n_norm = 2.0 * (coords[:, 0] - self._min_n) / (self._max_n - self._min_n) - 1.0
        e_norm = 2.0 * (coords[:, 1] - self._min_e) / (self._max_e - self._min_e) - 1.0
        norm_ne = np.column_stack([n_norm, e_norm]).astype(np.float32)
        cyclic = encode_cyclic_cell_uv(
            coords, harmonics=self.cyclic_cell_harmonics, spg_path=self.spg_path
        )
        aux = np.column_stack([norm_ne, cyclic]).astype(np.float32)
        return norm_ne, aux

    def _fit_impl(self, coords: np.ndarray, shifts: np.ndarray) -> None:
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        min_e, max_e, min_n, max_n, _, _ = get_spg_grid_bounds(self.spg_path)
        self._min_n, self._max_n = min_n, max_n
        self._min_e, self._max_e = min_e, max_e

        norm_ne, aux = self._encode_inputs(coords)
        targets = shifts.astype(np.float32)
        self.net = _KPlanesNet(
            n_bins=self.n_bins,
            feature_dim=self.feature_dim,
            aux_dim=aux.shape[1],
            hidden_dim=self.hidden_dim,
        ).to(self.device)
        train_torch_regressor(
            net=self.net,
            forward_fn=self.net,
            inputs=(norm_ne, aux),
            targets=targets,
            batch_size=self.batch_size,
            lr=self.lr,
            weight_decay=self.weight_decay,
            max_epochs=self.max_epochs,
            patience=self.patience,
            val_fraction=self.val_fraction,
            loss_type=self.loss_type,
            device=self.device,
            random_state=self.random_state,
        )

    def _predict_impl(self, coords: np.ndarray) -> np.ndarray:
        if self.net is None:
            raise RuntimeError("Model is not fitted yet.")
        norm_ne, aux = self._encode_inputs(coords)
        self.net.eval()
        preds: list[np.ndarray] = []
        chunk_size = 16384
        with torch.no_grad():
            for i in range(0, len(norm_ne), chunk_size):
                bn = torch.from_numpy(norm_ne[i : i + chunk_size]).to(self.device)
                ba = torch.from_numpy(aux[i : i + chunk_size]).to(self.device)
                preds.append(self.net(bn, ba).cpu().numpy())
        return np.vstack(preds).astype(np.float64)

    def _runtime_dependencies(self) -> tuple[str, ...]:
        return ("torch", "numpy")

    def _grid_lookup_bytes(self) -> int:
        if self.net is None:
            return 0
        return int(
            self.net.plane_n.numel() * self.net.plane_n.element_size()
            + self.net.plane_e.numel() * self.net.plane_e.element_size()
        )

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        if self.net is not None:
            state["net"] = copy.deepcopy(self.net).cpu()
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            self.device = "cpu"
        if self.net is not None:
            self.net = self.net.to(self.device)
            self.net.eval()
