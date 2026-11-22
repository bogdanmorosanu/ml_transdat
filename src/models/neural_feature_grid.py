"""Learnable multi-resolution 2D feature grid + compact MLP shift regressor."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from pytransdat.spg_reader import SpgReader
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
        reader = SpgReader(self.spg_path) if self.spg_path is not None else SpgReader()
        min_e, max_e, min_n, max_n, step_e, step_n = reader.shift_bounds
        self._min_n = float(min_n)
        self._max_n = float(max_n)
        self._min_e = float(min_e)
        self._max_e = float(max_e)
        self._step_n = float(step_n)
        self._step_e = float(step_e)

    def _encode_inputs(self, coords: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(grid_coords [-1, 1] as (e, n), aux_feats)`` in float32."""
        n_arr = coords[:, 0]
        e_arr = coords[:, 1]
        n_norm = 2.0 * (n_arr - self._min_n) / (self._max_n - self._min_n) - 1.0
        e_norm = 2.0 * (e_arr - self._min_e) / (self._max_e - self._min_e) - 1.0
        # grid_sample expects (x=Easting, y=Northing) in [-1, 1]
        grid_coords = np.column_stack([e_norm, n_norm]).astype(np.float32)

        r_idx = np.floor((n_arr - self._min_n) / self._step_n)
        c_idx = np.floor((e_arr - self._min_e) / self._step_e)
        u = (n_arr - (self._min_n + r_idx * self._step_n)) / self._step_n
        v = (e_arr - (self._min_e + c_idx * self._step_e)) / self._step_e

        aux_cols: list[np.ndarray] = [n_norm, e_norm]
        for k in range(1, self.cyclic_cell_harmonics + 1):
            ang_u = 2.0 * np.pi * k * u
            ang_v = 2.0 * np.pi * k * v
            aux_cols.extend([np.sin(ang_u), np.cos(ang_u), np.sin(ang_v), np.cos(ang_v)])
        aux_feats = np.column_stack(aux_cols).astype(np.float32)
        return grid_coords, aux_feats

    def _build_criterion(self):
        if self.loss_type == "mse":
            return nn.MSELoss()
        if self.loss_type == "huber":
            return nn.SmoothL1Loss(beta=0.005)
        if self.loss_type == "charbonnier":
            return lambda p, y: torch.mean(torch.sqrt((p - y) ** 2 + 1e-8) - 1e-4)
        raise ValueError(f"Unsupported loss_type '{self.loss_type}'.")

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

        n_samples = len(grid_coords)
        if self.val_fraction > 0.0 and n_samples >= 10:
            perm = np.random.RandomState(self.random_state).permutation(n_samples)
            val_size = max(int(self.val_fraction * n_samples), 1)
            val_idx = perm[:val_size]
            train_idx = perm[val_size:]
        else:
            train_idx = np.arange(n_samples)
            val_idx = None

        train_gc = torch.from_numpy(grid_coords[train_idx])
        train_aux = torch.from_numpy(aux_feats[train_idx])
        train_y = torch.from_numpy(targets[train_idx])
        loader = DataLoader(
            TensorDataset(train_gc, train_aux, train_y),
            batch_size=self.batch_size,
            shuffle=True,
        )

        if val_idx is not None:
            val_gc_t = torch.from_numpy(grid_coords[val_idx]).to(self.device)
            val_aux_t = torch.from_numpy(aux_feats[val_idx]).to(self.device)
            val_y_t = torch.from_numpy(targets[val_idx]).to(self.device)

        optimizer = torch.optim.AdamW(
            self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(self.max_epochs, 1), eta_min=1e-6
        )
        criterion = self._build_criterion()

        best_loss = float("inf")
        best_state: dict[str, torch.Tensor] | None = None
        no_improve = 0

        for _ in range(self.max_epochs):
            self.net.train()
            epoch_loss = 0.0
            n_batches = 0
            for bgc, baux, by in loader:
                bgc = bgc.to(self.device)
                baux = baux.to(self.device)
                by = by.to(self.device)
                optimizer.zero_grad()
                pred = self.net(bgc, baux)
                loss = criterion(pred, by)
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.item())
                n_batches += 1
            scheduler.step()

            if val_idx is not None:
                self.net.eval()
                with torch.no_grad():
                    mon_loss = float(criterion(self.net(val_gc_t, val_aux_t), val_y_t).item())
            else:
                mon_loss = epoch_loss / max(n_batches, 1)

            if mon_loss < best_loss:
                best_loss = mon_loss
                best_state = {k: v.cpu().clone() for k, v in self.net.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1

            if no_improve >= self.patience:
                break

        if best_state is not None:
            self.net.load_state_dict({k: v.to(self.device) for k, v in best_state.items()})

        if self.lbfgs_steps > 0:
            full_gc = train_gc.to(self.device)
            full_aux = train_aux.to(self.device)
            full_y = train_y.to(self.device)
            self.net.eval()
            with torch.no_grad():
                pre_loss = float(criterion(self.net(full_gc, full_aux), full_y).item())
            pre_state = {k: v.cpu().clone() for k, v in self.net.state_dict().items()}
            lbfgs = torch.optim.LBFGS(
                self.net.parameters(),
                lr=0.5,
                max_iter=self.lbfgs_steps,
                history_size=30,
                line_search_fn="strong_wolfe",
            )
            self.net.train()

            def _closure() -> torch.Tensor:
                lbfgs.zero_grad()
                out = self.net(full_gc, full_aux)
                l_val = criterion(out, full_y)
                l_val.backward()
                return l_val

            lbfgs.step(_closure)
            self.net.eval()
            with torch.no_grad():
                post_loss = float(criterion(self.net(full_gc, full_aux), full_y).item())
            if not np.isfinite(post_loss) or post_loss > pre_loss:
                self.net.load_state_dict({k: v.to(self.device) for k, v in pre_state.items()})

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
