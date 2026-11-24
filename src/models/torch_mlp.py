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

from src.models._features import SpatialFeatureEncoder
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

    Supports Random Fourier Features (RFF), configurable residual depth and layer
    widths, smooth/periodic activations (SiLU, GELU, Mish, ReLU, SIREN), and
    optional cyclic cell-phase and neighbor features via :class:`SpatialFeatureEncoder`.
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
        lr_scheduler: str = "cosine",
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

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.use_neighbor_features = bool(use_neighbor_features)
        self.neighbor_feature_mode = str(neighbor_feature_mode)
        self.neighbor_features_path = str(neighbor_features_path)
        self.scale_neighbor_features = bool(scale_neighbor_features)
        self.use_cell_center_distance = bool(use_cell_center_distance)
        self.use_cyclic_cell_pos = bool(use_cyclic_cell_pos)
        self.use_hermite_weights = bool(use_hermite_weights)

        self._feature_encoder = SpatialFeatureEncoder(
            normalize_inputs=self.normalize_inputs,
            use_fourier_features=self.use_fourier_features,
            fourier_scale=self.fourier_scale,
            fourier_scales=self.fourier_scales,
            num_fourier_frequencies=self.num_fourier_frequencies,
            random_state=self.random_state,
            use_neighbor_features=self.use_neighbor_features,
            neighbor_feature_mode=self.neighbor_feature_mode,
            neighbor_features_path=self.neighbor_features_path,
            scale_neighbor_features=self.scale_neighbor_features,
            use_hermite_weights=self.use_hermite_weights,
            use_cell_center_distance=self.use_cell_center_distance,
            use_cyclic_cell_pos=self.use_cyclic_cell_pos,
            cyclic_cell_harmonics=self.cyclic_cell_harmonics,
        )
        self.net: TorchMLPNet | None = None

    def _transform_features(self, coords: np.ndarray) -> np.ndarray:
        """Scale coordinates to [-1, 1] and apply configured spatial encodings."""
        return self._feature_encoder.transform(coords)

    def _fit_impl(self, coords: np.ndarray, shifts: np.ndarray) -> None:
        """Train the deep PyTorch residual network on (coords, shifts)."""
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_state)

        x_feats = self._feature_encoder.fit_transform(coords).astype(np.float32)
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

        x_feats = self._feature_encoder.transform(coords).astype(np.float32)
        self.net.eval()

        preds_list: list[np.ndarray] = []
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
