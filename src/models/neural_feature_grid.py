"""Learnable multi-resolution 2D feature grid + compact MLP shift regressor."""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from src.geometry.cell_uv import encode_cyclic_cell_uv, get_spg_grid_bounds
from src.models._torch_trainer import train_torch_regressor
from src.models.base import BaseGridShiftModel


class _FeatureGridMLPNet(nn.Module):
    """Multi-resolution bilinearly sampled 2D feature grids decoded by a tiny MLP."""

    def __init__(
        self,
        grid_resolutions: tuple[tuple[int, int], ...],
        feature_dim: int,
        aux_dim: int,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        self.grids = nn.ParameterList(
            [
                nn.Parameter(torch.zeros(1, feature_dim, r, c))
                for (r, c) in grid_resolutions
            ]
        )
        for g in self.grids:
            nn.init.normal_(g, mean=0.0, std=1e-3)

        in_dim = len(grid_resolutions) * feature_dim + aux_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(
        self, grid_coords: torch.Tensor, aux_feats: torch.Tensor
    ) -> torch.Tensor:
        """Evaluate bilinearly sampled grid features + ``aux_feats`` through the MLP.

        :param grid_coords: Normalized coordinates in ``[-1, 1]`` ordered ``(e_norm, n_norm)``, shape ``(B, 2)``.
        :param aux_feats: Global + cyclic intra-cell features, shape ``(B, aux_dim)``.
        :return: Predicted shifts ``(B, 2)``.
        """
        sample_grid = grid_coords.view(1, -1, 1, 2)
        sampled: list[torch.Tensor] = []
        for g in self.grids:
            feat = F.grid_sample(
                g, sample_grid, mode="bilinear", padding_mode="border", align_corners=True
            )
            sampled.append(feat.squeeze(0).squeeze(-1).transpose(0, 1))
        x = torch.cat([*sampled, aux_feats], dim=1)
        return self.mlp(x)


class NeuralFeatureGridShiftModel(BaseGridShiftModel):
    """Instant-NGP-inspired learnable multi-resolution 2D feature grid + tiny MLP.

    Aligns its coarse feature grid directly to Romania's 11 km ``.spg`` lattice so
    bilinear interpolation naturally captures ``C^0`` continuous values with sharp
    ``C^1`` slope bends across cell boundaries while remaining smooth inside cells.
    """

    def __init__(
        self,
        grid_resolutions: tuple[tuple[int, int], ...] = ((53, 72), (105, 143)),
        feature_dim: int = 8,
        hidden_dim: int = 64,
        cyclic_cell_harmonics: int = 2,
        batch_size: int = 512,
        lr: float = 3e-3,
        weight_decay: float = 1e-6,
        max_epochs: int = 200,
        patience: int = 30,
        val_fraction: float = 0.0,
        loss_type: str = "mse",
        lbfgs_steps: int = 0,
        device: str | None = None,
        random_state: int = 42,
        spg_path: str | None = None,
    ) -> None:
        super().__init__()
        self.grid_resolutions = tuple((int(r), int(c)) for r, c in grid_resolutions)
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
        self.lbfgs_steps = int(lbfgs_steps)
        self.random_state = int(random_state)
        self.spg_path = spg_path
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self._min_n: float = 0.0
        self._max_n: float = 1.0
        self._min_e: float = 0.0
        self._max_e: float = 1.0
        self._step_n: float = 11000.0
        self._step_e: float = 11000.0
        self.net: _FeatureGridMLPNet | None = None

    def _init_bounds(self) -> None:
        min_e, max_e, min_n, max_n, step_e, step_n = get_spg_grid_bounds(self.spg_path)
        self._min_n = min_n
        self._max_n = max_n
        self._min_e = min_e
        self._max_e = max_e
        self._step_n = step_n
        self._step_e = step_e

    def _encode_inputs(self, coords: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(grid_coords [-1, 1] as (e, n), aux_feats)`` in float32."""
        n_arr = coords[:, 0]
        e_arr = coords[:, 1]
        n_norm = 2.0 * (n_arr - self._min_n) / (self._max_n - self._min_n) - 1.0
        e_norm = 2.0 * (e_arr - self._min_e) / (self._max_e - self._min_e) - 1.0
        # grid_sample expects (x=Easting, y=Northing) in [-1, 1]
        grid_coords = np.column_stack([e_norm, n_norm]).astype(np.float32)

        cyclic = encode_cyclic_cell_uv(
            coords, harmonics=self.cyclic_cell_harmonics, spg_path=self.spg_path
        )
        aux_feats = np.column_stack([n_norm, e_norm, cyclic]).astype(np.float32)
        return grid_coords, aux_feats

    def _fit_impl(self, coords: np.ndarray, shifts: np.ndarray) -> None:
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_state)

        self._init_bounds()
        grid_coords, aux_feats = self._encode_inputs(coords)
        targets = shifts.astype(np.float32)

        self.net = _FeatureGridMLPNet(
            grid_resolutions=self.grid_resolutions,
            feature_dim=self.feature_dim,
            aux_dim=aux_feats.shape[1],
            hidden_dim=self.hidden_dim,
        ).to(self.device)

        train_torch_regressor(
            net=self.net,
            forward_fn=self.net,
            inputs=(grid_coords, aux_feats),
            targets=targets,
            batch_size=self.batch_size,
            lr=self.lr,
            weight_decay=self.weight_decay,
            max_epochs=self.max_epochs,
            patience=self.patience,
            lr_scheduler="cosine",
            val_fraction=self.val_fraction,
            loss_type=self.loss_type,
            lbfgs_steps=self.lbfgs_steps,
            lbfgs_history_size=30,
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
        if self.net is None:
            raise RuntimeError("Model is not fitted yet.")
        grid_coords, aux_feats = self._encode_inputs(coords)
        self.net.eval()
        preds_list: list[np.ndarray] = []
        chunk_size = 16384
        with torch.no_grad():
            for i in range(0, len(grid_coords), chunk_size):
                bgc = torch.from_numpy(grid_coords[i : i + chunk_size]).to(self.device)
                baux = torch.from_numpy(aux_feats[i : i + chunk_size]).to(self.device)
                preds_list.append(self.net(bgc, baux).cpu().numpy())
        return np.vstack(preds_list).astype(np.float64)

    def _runtime_dependencies(self) -> tuple[str, ...]:
        return ("torch", "numpy")

    def _grid_lookup_bytes(self) -> int:
        if self.net is None:
            return 0
        return int(sum(p.numel() * p.element_size() for p in self.net.grids))
