"""PyTorch Deep Residual Neural Network regressor for 2D geodetic distortion grid shifts."""

from __future__ import annotations

import copy
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from src.geometry.cell_uv import compute_spg_cell_uv, encode_cyclic_cell_uv
from src.models._torch_trainer import train_torch_regressor
from src.models.base import BaseGridShiftModel

logger = logging.getLogger(__name__)


class SineActivation(nn.Module):
    """SIREN periodic sine activation with configurable frequency scaling ``omega_0``."""

    def __init__(self, omega_0: float = 30.0) -> None:
        super().__init__()
        self.omega_0 = float(omega_0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.omega_0 * x)


class ResidualBlock(nn.Module):
    """Residual dense block with optional LayerNorm and configurable activation."""

    def __init__(
        self,
        hidden_dim: int,
        activation_fn: Callable[[], nn.Module] = nn.SiLU,
        dropout: float = 0.0,
        use_layernorm: bool = True,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(hidden_dim, hidden_dim)]
        if use_layernorm:
            layers.append(nn.LayerNorm(hidden_dim))
        layers.append(activation_fn())
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dim, hidden_dim))
        if use_layernorm:
            layers.append(nn.LayerNorm(hidden_dim))
        self.block = nn.Sequential(*layers)
        self.act = activation_fn()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.block(x))


class TorchMLPNet(nn.Module):
    """Deep PyTorch feedforward / residual network for spatial shift regression."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int = 2,
        hidden_dim: int = 256,
        num_res_blocks: int = 3,
        activation: str = "silu",
        dropout: float = 0.0,
        siren_omega_0: float = 30.0,
    ) -> None:
        super().__init__()
        act_name = activation.lower()
        act_map: dict[str, Callable[[], nn.Module]] = {
            "relu": nn.ReLU,
            "silu": nn.SiLU,
            "gelu": nn.GELU,
            "tanh": nn.Tanh,
            "mish": nn.Mish,
            "siren": lambda: SineActivation(omega_0=siren_omega_0),
        }
        if act_name not in act_map:
            raise ValueError(
                f"Unsupported activation '{activation}'. Choose from {list(act_map.keys())}"
            )

        act_fn = act_map[act_name]
        is_siren = act_name == "siren"
        use_norm = not is_siren

        proj_layers: list[nn.Module] = [nn.Linear(in_dim, hidden_dim)]
        if use_norm:
            proj_layers.append(nn.LayerNorm(hidden_dim))
        proj_layers.append(act_fn())
        self.input_proj = nn.Sequential(*proj_layers)

        blocks: list[nn.Module] = []
        for _ in range(num_res_blocks):
            blocks.append(
                ResidualBlock(
                    hidden_dim,
                    activation_fn=act_fn,
                    dropout=dropout,
                    use_layernorm=use_norm,
                )
            )
        self.res_blocks = nn.Sequential(*blocks)

        self.head = nn.Linear(hidden_dim, out_dim)

        if is_siren:
            self._init_siren_weights(siren_omega_0)

    def _init_siren_weights(self, omega_0: float) -> None:
        """Apply principled SIREN uniform weight initialization (Sitzmann et al.)."""
        with torch.no_grad():
            first_linear = self.input_proj[0]
            if isinstance(first_linear, nn.Linear):
                bound = 1.0 / first_linear.in_features
                first_linear.weight.uniform_(-bound, bound)
                if first_linear.bias is not None:
                    first_linear.bias.zero_()
            for module in self.res_blocks.modules():
                if isinstance(module, nn.Linear):
                    bound = float(np.sqrt(6.0 / module.in_features)) / omega_0
                    module.weight.uniform_(-bound, bound)
                    if module.bias is not None:
                        module.bias.zero_()
            bound_head = float(np.sqrt(6.0 / self.head.in_features)) / omega_0
            self.head.weight.uniform_(-bound_head, bound_head)
            if self.head.bias is not None:
                self.head.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        h = self.res_blocks(h)
        return self.head(h)


class TorchMLPGridShiftModel(BaseGridShiftModel):
    """Deep PyTorch Neural Network Regressor for 2D planar coordinate distortion shifts.

    Approximates the continuous mapping:
        (st70_n, st70_e) -> (shift_n, shift_e)

    Supports:
    - Random Fourier Features (RFF) positional encoding for anti-aliasing / resolving spectral bias
    - Configurable residual depth and layer widths
    - Smooth activations (SiLU, GELU, Mish, ReLU)
    - Full PyTorch GPU / CUDA acceleration
    - Optional integration with continuous cell neighbor features
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_res_blocks: int = 3,
        activation: str = "silu",
        dropout: float = 0.0,
        normalize_inputs: bool = True,
        use_fourier_features: bool = True,
        fourier_scale: float = 3.0,
        num_fourier_frequencies: int = 64,
        batch_size: int = 256,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        max_epochs: int = 250,
        patience: int = 25,
        lr_scheduler: str = "cosine",  # "cosine" or "plateau"
        device: str | None = None,
        random_state: int = 42,
        use_neighbor_features: bool = False,
        neighbor_feature_mode: str = "all",
        neighbor_features_path: str | Path = "data/features/cell_neighbor_features.parquet",
        scale_neighbor_features: bool = True,
        use_cell_center_distance: bool = False,
        use_cyclic_cell_pos: bool = False,
        use_hermite_weights: bool = False,
        cyclic_cell_harmonics: int = 1,
        fourier_scales: tuple[float, ...] | None = None,
        val_fraction: float = 0.1,
        loss_type: str = "mse",
        lbfgs_steps: int = 0,
        siren_omega_0: float = 30.0,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_res_blocks = int(num_res_blocks)
        self.activation = str(activation).lower()
        self.dropout = float(dropout)
        self.normalize_inputs = bool(normalize_inputs)
        self.use_fourier_features = bool(use_fourier_features)
        self.fourier_scale = float(fourier_scale)
        self.fourier_scales = (
            tuple(float(s) for s in fourier_scales) if fourier_scales is not None else None
        )
        self.num_fourier_frequencies = int(num_fourier_frequencies)
        self.batch_size = int(batch_size)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.max_epochs = int(max_epochs)
        self.patience = int(patience)
        self.lr_scheduler = str(lr_scheduler).lower()
        self.random_state = int(random_state)
        self.cyclic_cell_harmonics = int(cyclic_cell_harmonics)
        self.val_fraction = float(val_fraction)
        self.loss_type = str(loss_type).lower()
        self.lbfgs_steps = int(lbfgs_steps)
        self.siren_omega_0 = float(siren_omega_0)

        # Device selection: prefer CUDA if available
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Neighbor feature options
        self.use_neighbor_features = bool(use_neighbor_features)
        self.neighbor_feature_mode = str(neighbor_feature_mode)
        self.neighbor_features_path = str(neighbor_features_path)
        self.scale_neighbor_features = bool(scale_neighbor_features)
        self.use_cell_center_distance = bool(use_cell_center_distance)
        self.use_cyclic_cell_pos = bool(use_cyclic_cell_pos)
        self.use_hermite_weights = bool(use_hermite_weights)

        self._neighbor_extractor: Any = None
        self._neighbor_col_indices: np.ndarray | None = None
        self._neighbor_mean: np.ndarray | None = None
        self._neighbor_std: np.ndarray | None = None

        self._min_coords: np.ndarray | None = None
        self._max_coords: np.ndarray | None = None
        self._fourier_basis: np.ndarray | None = None
        self.net: TorchMLPNet | None = None

    def _ensure_neighbor_extractor(self) -> None:
        if self._neighbor_extractor is None:
            from src.geometry.neighbor_features import (
                NEIGHBOR_FEATURE_COLS,
                CellNeighborFeatureExtractor,
            )

            self._neighbor_extractor = CellNeighborFeatureExtractor(
                self.neighbor_features_path
            )
            if self.neighbor_feature_mode == "slopes":
                self._neighbor_col_indices = np.array(
                    [i for i, c in enumerate(NEIGHBOR_FEATURE_COLS) if "slope" in c],
                    dtype=np.int64,
                )
            else:
                self._neighbor_col_indices = np.arange(
                    len(NEIGHBOR_FEATURE_COLS), dtype=np.int64
                )

    def _compute_cell_uv(self, coords: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return normalized intra-cell (u, v) in [0, 1) for Northing and Easting."""
        return compute_spg_cell_uv(coords)

    def _compute_hermite_weights(self, coords: np.ndarray) -> np.ndarray:
        """Compute 2D cubic Hermite basis weights aligned with selected neighbor columns."""
        from src.geometry.neighbor_features import NEIGHBOR_FEATURE_COLS

        u, v = self._compute_cell_uv(coords)
        h0_u = (1.0 + 2.0 * u) * (1.0 - u) ** 2
        h1_u = (u ** 2) * (3.0 - 2.0 * u)
        h10_u = u * (1.0 - u) ** 2
        h11_u = (u ** 2) * (u - 1.0)

        h0_v = (1.0 + 2.0 * v) * (1.0 - v) ** 2
        h1_v = (v ** 2) * (3.0 - 2.0 * v)
        h10_v = v * (1.0 - v) ** 2
        h11_v = (v ** 2) * (v - 1.0)

        corner_basis = {
            "bl": (h0_u, h10_u, h0_v, h10_v),
            "br": (h0_u, h10_u, h1_v, h11_v),
            "tl": (h1_u, h11_u, h0_v, h10_v),
            "tr": (h1_u, h11_u, h1_v, h11_v),
        }

        selected_cols = [NEIGHBOR_FEATURE_COLS[i] for i in self._neighbor_col_indices]
        weights = np.empty((len(coords), len(selected_cols)), dtype=np.float64)
        for j, col in enumerate(selected_cols):
            _, corner, feat = col.split("_", 2)
            hu, hdu, hv, hdv = corner_basis[corner]
            if feat in ("left_slope", "right_slope"):
                weights[:, j] = hu * hdv
            elif feat in ("bottom_slope", "top_slope"):
                weights[:, j] = hdu * hv
            else:  # twist
                weights[:, j] = hdu * hdv
        return weights

    def _extract_neighbor_features(self, coords: np.ndarray) -> np.ndarray:
        self._ensure_neighbor_extractor()
        feats = self._neighbor_extractor.lookup_points(coords[:, 0], coords[:, 1])
        feats = feats[:, self._neighbor_col_indices]
        return np.nan_to_num(feats, nan=0.0)

    def _compute_hermite_sum_features(
        self, coords: np.ndarray, scaled_all: np.ndarray
    ) -> np.ndarray:
        """Condense 40 scaled neighbor features into 6 smooth Hermite-summed terms."""
        weights = self._compute_hermite_weights(coords)
        weighted = scaled_all * weights
        # NEIGHBOR_FEATURE_COLS has 20 cols for dn then 20 cols for de:
        # 4 corners x 5 feats (left_slope=0, right_slope=1, bottom_slope=2, top_slope=3, twist=4)
        out = np.empty((len(coords), 6), dtype=np.float64)
        for ch_idx in range(2):
            base = ch_idx * 20
            block = weighted[:, base : base + 20].reshape(-1, 4, 5)
            dv_sum = np.sum(0.5 * (block[:, :, 0] + block[:, :, 1]), axis=1)
            du_sum = np.sum(0.5 * (block[:, :, 2] + block[:, :, 3]), axis=1)
            tw_sum = np.sum(block[:, :, 4], axis=1)
            out[:, ch_idx * 3 + 0] = du_sum
            out[:, ch_idx * 3 + 1] = dv_sum
            out[:, ch_idx * 3 + 2] = tw_sum
        return out

    def _transform_features(self, coords: np.ndarray) -> np.ndarray:
        """Scale coordinates to [-1, 1] and optionally apply Fourier positional encoding."""
        if self.normalize_inputs:
            if self._min_coords is None or self._max_coords is None:
                raise RuntimeError("Model fitted state missing normalization parameters.")
            diff = self._max_coords - self._min_coords
            diff[diff == 0.0] = 1.0
            x_norm = 2.0 * (coords - self._min_coords) / diff - 1.0
        else:
            x_norm = coords

        if not self.use_fourier_features:
            x_spatial = x_norm
        else:
            if self._fourier_basis is None:
                raise RuntimeError("Fourier projection matrix B is uninitialized.")
            proj = 2.0 * np.pi * (x_norm @ self._fourier_basis)
            x_spatial = np.concatenate([x_norm, np.sin(proj), np.cos(proj)], axis=1)

        if self.use_neighbor_features:
            neighbor_feats = self._extract_neighbor_features(coords)
            if self.scale_neighbor_features:
                if self._neighbor_mean is None or self._neighbor_std is None:
                    raise RuntimeError("Missing neighbor feature scaling parameters.")
                neighbor_feats = (neighbor_feats - self._neighbor_mean) / self._neighbor_std
            if self.neighbor_feature_mode == "hermite_sum":
                neighbor_feats = self._compute_hermite_sum_features(coords, neighbor_feats)
            elif self.use_hermite_weights:
                neighbor_feats = neighbor_feats * self._compute_hermite_weights(coords)
            x_spatial = np.concatenate([x_spatial, neighbor_feats], axis=1)

        if self.use_cell_center_distance:
            self._ensure_neighbor_extractor()
            center_dists = self._neighbor_extractor.lookup_center_distances(
                coords[:, 0], coords[:, 1]
            )
            x_spatial = np.concatenate([x_spatial, center_dists], axis=1)

        if self.use_cyclic_cell_pos:
            cyclic_pos = encode_cyclic_cell_uv(
                coords,
                harmonics=self.cyclic_cell_harmonics,
                include_cross_terms=self.cyclic_cell_harmonics > 1,
            )
            x_spatial = np.concatenate([x_spatial, cyclic_pos], axis=1)

        return x_spatial

    def _fit_impl(self, coords: np.ndarray, shifts: np.ndarray) -> None:
        """Train the deep PyTorch residual network on (coords, shifts)."""
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_state)

        if self.normalize_inputs:
            self._min_coords = np.min(coords, axis=0)
            self._max_coords = np.max(coords, axis=0)

        if self.use_fourier_features:
            rng = np.random.RandomState(self.random_state)
            if self.fourier_scales is not None and len(self.fourier_scales) > 0:
                n_bands = len(self.fourier_scales)
                base_n = self.num_fourier_frequencies // n_bands
                rem = self.num_fourier_frequencies % n_bands
                bands: list[np.ndarray] = []
                for idx, sc in enumerate(self.fourier_scales):
                    n_k = base_n + (1 if idx < rem else 0)
                    if n_k > 0:
                        bands.append(rng.randn(2, n_k) * sc)
                self._fourier_basis = np.hstack(bands)
            else:
                self._fourier_basis = (
                    rng.randn(2, self.num_fourier_frequencies) * self.fourier_scale
                )

        if self.use_neighbor_features and self.scale_neighbor_features:
            raw_neighbor = self._extract_neighbor_features(coords)
            self._neighbor_mean = np.mean(raw_neighbor, axis=0)
            std = np.std(raw_neighbor, axis=0)
            std[std == 0.0] = 1.0
            self._neighbor_std = std

        x_feats = self._transform_features(coords).astype(np.float32)
        y_targets = shifts.astype(np.float32)

        in_dim = x_feats.shape[1]
        self.net = TorchMLPNet(
            in_dim=in_dim,
            out_dim=2,
            hidden_dim=self.hidden_dim,
            num_res_blocks=self.num_res_blocks,
            activation=self.activation,
            dropout=self.dropout,
            siren_omega_0=self.siren_omega_0,
        ).to(self.device)

        train_torch_regressor(
            net=self.net,
            forward_fn=self.net,
            inputs=(x_feats,),
            targets=y_targets,
            batch_size=self.batch_size,
            lr=self.lr,
            weight_decay=self.weight_decay,
            max_epochs=self.max_epochs,
            patience=self.patience,
            lr_scheduler=self.lr_scheduler,
            val_fraction=self.val_fraction,
            loss_type=self.loss_type,
            lbfgs_steps=self.lbfgs_steps,
            lbfgs_history_size=50,
            device=self.device,
            random_state=self.random_state,
        )

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_neighbor_extractor"] = None
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

    def _predict_impl(self, coords: np.ndarray) -> np.ndarray:
        """Predict shift_n and shift_e for given query coordinates."""
        if self.net is None:
            raise RuntimeError("Model is not fitted yet.")

        x_feats = self._transform_features(coords).astype(np.float32)
        self.net.eval()
        
        preds_list: list[np.ndarray] = []
        # Predict in chunks to manage memory
        chunk_size = 16384
        with torch.no_grad():
            for i in range(0, len(x_feats), chunk_size):
                bx = torch.from_numpy(x_feats[i : i + chunk_size]).to(self.device)
                out = self.net(bx)
                preds_list.append(out.cpu().numpy())

        return np.vstack(preds_list).astype(np.float64)

    def _runtime_dependencies(self) -> tuple[str, ...]:
        return ("torch", "numpy")

    def _grid_lookup_bytes(self) -> int:
        return 0

    @property
    def total_parameters(self) -> int:
        if self.net is None:
            return 0
        return sum(p.numel() for p in self.net.parameters() if p.requires_grad)
