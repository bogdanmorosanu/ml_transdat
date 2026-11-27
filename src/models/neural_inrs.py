"""Modern Implicit Neural Representation (INR) shift regressors (0 MB lookup).

Implements four PyTorch coordinate-network families evaluated on Stage-2
millimeter-scale residuals:
- ``KANGridShiftModel``: Pure-PyTorch RBF-KAN and Fourier-KAN layers.
- ``WaveletINRGridShiftModel``: Complex Gabor (WIRE) and variable-periodic (FINER) INRs.
- ``MFNGridShiftModel``: Multiplicative Filter Networks (GaborNet / FourierNet).
- ``SpatialMoEGridShiftModel``: Spatial Mixture-of-Experts and FBPINN overlapping subdomains.
"""

from __future__ import annotations

import copy
import math
from typing import Any

import numpy as np
import torch
from torch import nn

from src.models._features import SpatialFeatureEncoder
from src.models._torch_trainer import train_torch_regressor
from src.models.base import BaseGridShiftModel


class _BaseTorchINRModel(BaseGridShiftModel):
    """Shared device, feature-encoder, and pickling boilerplate for PyTorch INRs."""

    net: nn.Module | None
    encoder: SpatialFeatureEncoder | None
    device: str

    def _init_device(self, device: str | None) -> str:
        if device is None:
            return "cuda" if torch.cuda.is_available() else "cpu"
        return str(device)

    def _predict_chunked(self, feats: np.ndarray, chunk_size: int = 16384) -> np.ndarray:
        if self.net is None:
            raise RuntimeError("Model is not fitted yet.")
        self.net.eval()
        preds: list[np.ndarray] = []
        with torch.no_grad():
            for i in range(0, len(feats), chunk_size):
                bx = torch.from_numpy(feats[i : i + chunk_size]).to(self.device)
                preds.append(self.net(bx).cpu().numpy())
        return np.vstack(preds).astype(np.float64)

    def _runtime_dependencies(self) -> tuple[str, ...]:
        return ("torch", "numpy")

    def _grid_lookup_bytes(self) -> int:
        return 0

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
# V2-1: Kolmogorov-Arnold Networks (RBF-KAN & Fourier-KAN)
# ---------------------------------------------------------------------------


class _RBFKANLayer(nn.Module):
    """Fast pure-PyTorch RBF-KAN layer with SiLU residual skip connection."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_basis: int = 8,
        grid_min: float = -2.0,
        grid_max: float = 2.0,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        grid = torch.linspace(grid_min, grid_max, num_basis)
        self.register_buffer("grid", grid)
        self.inv_denom = float(num_basis - 1) / float(grid_max - grid_min)
        self.spline_weight = nn.Parameter(
            torch.randn(out_dim, in_dim, num_basis) * (1.0 / math.sqrt(in_dim * num_basis))
        )
        self.base_linear = nn.Linear(in_dim, out_dim)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_n = torch.tanh(self.norm(x)) * 2.0
        rbf = torch.exp(-(((x_n.unsqueeze(-1) - self.grid) * self.inv_denom) ** 2))
        spline_out = torch.einsum("bik,oik->bo", rbf, self.spline_weight)
        return spline_out + self.base_linear(self.act(x))


class _FourierKANLayer(nn.Module):
    """Fourier-series KAN layer with learnable per-edge harmonic weights."""

    def __init__(self, in_dim: int, out_dim: int, num_basis: int = 8) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        freqs = torch.arange(1, num_basis + 1, dtype=torch.float32)
        self.register_buffer("freqs", freqs)
        scale = 1.0 / (math.sqrt(in_dim) * math.sqrt(num_basis))
        self.cos_weight = nn.Parameter(torch.randn(out_dim, in_dim, num_basis) * scale)
        self.sin_weight = nn.Parameter(torch.randn(out_dim, in_dim, num_basis) * scale)
        self.bias = nn.Parameter(torch.zeros(out_dim))
        self.base_linear = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_n = self.norm(x)
        ang = x_n.unsqueeze(-1) * self.freqs
        c_out = torch.einsum("bik,oik->bo", torch.cos(ang), self.cos_weight)
        s_out = torch.einsum("bik,oik->bo", torch.sin(ang), self.sin_weight)
        return c_out + s_out + self.bias + self.base_linear(x)


class _KANNet(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_basis: int,
        basis: str,
    ) -> None:
        super().__init__()
        layer_cls = _RBFKANLayer if basis == "rbf" else _FourierKANLayer
        layers: list[nn.Module] = []
        curr = in_dim
        for _ in range(num_layers):
            layers.append(layer_cls(curr, hidden_dim, num_basis=num_basis))
            curr = hidden_dim
        layers.append(nn.Linear(curr, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class KANGridShiftModel(_BaseTorchINRModel):
    """Kolmogorov-Arnold Network (RBF-KAN or Fourier-KAN) 2D shift regressor."""

    def __init__(
        self,
        basis: str = "rbf",
        hidden_dim: int = 64,
        num_layers: int = 3,
        num_basis: int = 8,
        use_fourier_features: bool = True,
        fourier_scales: tuple[float, ...] = (1.0, 3.0, 10.0),
        num_fourier_frequencies: int = 48,
        use_cyclic_cell_pos: bool = True,
        cyclic_cell_harmonics: int = 2,
        batch_size: int = 1024,
        lr: float = 2e-3,
        weight_decay: float = 1e-5,
        max_epochs: int = 150,
        patience: int = 25,
        val_fraction: float = 0.0,
        loss_type: str = "huber",
        device: str | None = None,
        random_state: int = 42,
    ) -> None:
        super().__init__()
        b_norm = basis.lower()
        if b_norm not in ("rbf", "fourier"):
            raise ValueError(f"Unsupported basis='{basis}'. Expected 'rbf' or 'fourier'.")
        self.basis = b_norm
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.num_basis = int(num_basis)
        self.use_fourier_features = bool(use_fourier_features)
        self.fourier_scales = tuple(float(s) for s in fourier_scales)
        self.num_fourier_frequencies = int(num_fourier_frequencies)
        self.use_cyclic_cell_pos = bool(use_cyclic_cell_pos)
        self.cyclic_cell_harmonics = int(cyclic_cell_harmonics)
        self.batch_size = int(batch_size)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.max_epochs = int(max_epochs)
        self.patience = int(patience)
        self.val_fraction = float(val_fraction)
        self.loss_type = str(loss_type).lower()
        self.random_state = int(random_state)
        self.device = self._init_device(device)

        self.encoder: SpatialFeatureEncoder | None = None
        self.net: nn.Module | None = None

    def _fit_impl(self, coords: np.ndarray, shifts: np.ndarray) -> None:
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        self.encoder = SpatialFeatureEncoder(
            normalize_inputs=True,
            use_fourier_features=self.use_fourier_features,
            fourier_scales=self.fourier_scales,
            num_fourier_frequencies=self.num_fourier_frequencies,
            use_cyclic_cell_pos=self.use_cyclic_cell_pos,
            cyclic_cell_harmonics=self.cyclic_cell_harmonics,
            random_state=self.random_state,
        )
        feats = self.encoder.fit_transform(coords).astype(np.float32)
        targets = shifts.astype(np.float32)
        self.net = _KANNet(
            in_dim=feats.shape[1],
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            num_basis=self.num_basis,
            basis=self.basis,
        ).to(self.device)
        train_torch_regressor(
            net=self.net,
            forward_fn=self.net,
            inputs=(feats,),
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
        if self.encoder is None:
            raise RuntimeError("Model is not fitted yet.")
        feats = self.encoder.transform(coords).astype(np.float32)
        return self._predict_chunked(feats)


# ---------------------------------------------------------------------------
# V2-2: Wavelet INRs (WIRE: Complex Gabor) & Variable-Periodic (FINER)
# ---------------------------------------------------------------------------


class _ComplexGaborLayer(nn.Module):
    """Real-valued implementation of WIRE's complex Gabor wavelet layer."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        omega0: float = 10.0,
        scale0: float = 5.0,
        is_first: bool = False,
    ) -> None:
        super().__init__()
        self.omega0 = float(omega0)
        self.scale0 = float(scale0)
        self.in_dim = in_dim
        self.is_first = is_first
        self.lin_r = nn.Linear(in_dim, out_dim)
        self.lin_i = nn.Linear(in_dim, out_dim)
        bound = (1.0 / in_dim) if is_first else (math.sqrt(6.0 / in_dim) / max(omega0, 1.0))
        with torch.no_grad():
            self.lin_r.weight.uniform_(-bound, bound)
            self.lin_i.weight.uniform_(-bound, bound)

    def forward(self, xr: torch.Tensor, xi: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        if xi is None:
            zr = self.lin_r(xr)
            zi = self.lin_i(xr)
        else:
            zr = self.lin_r(xr) - self.lin_i(xi)
            zi = self.lin_r(xi) + self.lin_i(xr)

        # Complex Gabor: exp(j * omega0 * z - (scale0 * |z|)^2)
        # Note: exp(j*omega0*(zr + j*zi)) = exp(-omega0*zi) * (cos(omega0*zr) + j*sin(omega0*zr))
        mag_sq = zr * zr + zi * zi
        env = torch.exp(torch.clamp(-self.omega0 * zi - (self.scale0 * self.scale0) * mag_sq, min=-20.0, max=4.0))
        ang = self.omega0 * zr
        return env * torch.cos(ang), env * torch.sin(ang)


class _WIRENet(nn.Module):
    def __init__(
        self, in_dim: int, hidden_dim: int, num_layers: int, omega0: float, scale0: float
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(_ComplexGaborLayer(in_dim, hidden_dim, omega0=omega0, scale0=scale0, is_first=True))
        for _ in range(num_layers - 1):
            self.layers.append(_ComplexGaborLayer(hidden_dim, hidden_dim, omega0=omega0, scale0=scale0, is_first=False))
        self.out_r = nn.Linear(hidden_dim, 2)
        self.out_i = nn.Linear(hidden_dim, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xr: torch.Tensor = x
        xi: torch.Tensor | None = None
        for layer in self.layers:
            xr, xi = layer(xr, xi)
        assert xi is not None
        return self.out_r(xr) - self.out_i(xi)


class _FINERLayer(nn.Module):
    """Variable-periodic sin(omega0 * (|z| + 1) * z) layer."""

    def __init__(
        self, in_dim: int, out_dim: int, omega0: float = 10.0, is_first: bool = False
    ) -> None:
        super().__init__()
        self.omega0 = float(omega0)
        self.linear = nn.Linear(in_dim, out_dim)
        with torch.no_grad():
            if is_first:
                bound = 1.0 / in_dim
                self.linear.weight.uniform_(-bound, bound)
                self.linear.bias.uniform_(-1.0, 1.0)
            else:
                bound = math.sqrt(6.0 / in_dim) / max(omega0, 1.0)
                self.linear.weight.uniform_(-bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.linear(x)
        return torch.sin(self.omega0 * (torch.abs(z) + 1.0) * z)


class _FINERNet(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, num_layers: int, omega0: float) -> None:
        super().__init__()
        layers: list[nn.Module] = [_FINERLayer(in_dim, hidden_dim, omega0=omega0, is_first=True)]
        for _ in range(num_layers - 1):
            layers.append(_FINERLayer(hidden_dim, hidden_dim, omega0=omega0, is_first=False))
        layers.append(nn.Linear(hidden_dim, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class WaveletINRGridShiftModel(_BaseTorchINRModel):
    """Complex Gabor (WIRE) or Variable-Periodic (FINER) implicit neural representation."""

    def __init__(
        self,
        activation: str = "wire",
        hidden_dim: int = 128,
        num_layers: int = 4,
        omega0: float = 10.0,
        scale0: float = 5.0,
        use_cyclic_cell_pos: bool = True,
        cyclic_cell_harmonics: int = 2,
        batch_size: int = 1024,
        lr: float = 1e-3,
        weight_decay: float = 1e-6,
        max_epochs: int = 150,
        patience: int = 25,
        val_fraction: float = 0.0,
        loss_type: str = "huber",
        device: str | None = None,
        random_state: int = 42,
    ) -> None:
        super().__init__()
        act_norm = activation.lower()
        if act_norm not in ("wire", "finer"):
            raise ValueError(f"Unsupported activation='{activation}'. Expected 'wire' or 'finer'.")
        self.activation = act_norm
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.omega0 = float(omega0)
        self.scale0 = float(scale0)
        self.use_cyclic_cell_pos = bool(use_cyclic_cell_pos)
        self.cyclic_cell_harmonics = int(cyclic_cell_harmonics)
        self.batch_size = int(batch_size)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.max_epochs = int(max_epochs)
        self.patience = int(patience)
        self.val_fraction = float(val_fraction)
        self.loss_type = str(loss_type).lower()
        self.random_state = int(random_state)
        self.device = self._init_device(device)

        self.encoder: SpatialFeatureEncoder | None = None
        self.net: nn.Module | None = None

    def _fit_impl(self, coords: np.ndarray, shifts: np.ndarray) -> None:
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        self.encoder = SpatialFeatureEncoder(
            normalize_inputs=True,
            use_fourier_features=False,
            use_cyclic_cell_pos=self.use_cyclic_cell_pos,
            cyclic_cell_harmonics=self.cyclic_cell_harmonics,
            random_state=self.random_state,
        )
        feats = self.encoder.fit_transform(coords).astype(np.float32)
        targets = shifts.astype(np.float32)
        if self.activation == "wire":
            self.net = _WIRENet(
                in_dim=feats.shape[1],
                hidden_dim=self.hidden_dim,
                num_layers=self.num_layers,
                omega0=self.omega0,
                scale0=self.scale0,
            ).to(self.device)
        else:
            self.net = _FINERNet(
                in_dim=feats.shape[1],
                hidden_dim=self.hidden_dim,
                num_layers=self.num_layers,
                omega0=self.omega0,
            ).to(self.device)

        train_torch_regressor(
            net=self.net,
            forward_fn=self.net,
            inputs=(feats,),
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
        if self.encoder is None:
            raise RuntimeError("Model is not fitted yet.")
        feats = self.encoder.transform(coords).astype(np.float32)
        return self._predict_chunked(feats)


# ---------------------------------------------------------------------------
# V2-3: Multiplicative Filter Networks (GaborNet & FourierNet)
# ---------------------------------------------------------------------------


class _GaborFilter(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, input_scale: float) -> None:
        super().__init__()
        self.mu = nn.Parameter(torch.empty(hidden_dim, in_dim).uniform_(-1.0, 1.0))
        self.gamma = nn.Parameter(torch.ones(hidden_dim) * 4.0)
        self.linear = nn.Linear(in_dim, hidden_dim)
        with torch.no_grad():
            self.linear.weight.normal_(0.0, input_scale / math.sqrt(in_dim))
            self.linear.bias.uniform_(-math.pi, math.pi)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ||x - mu||^2 = |x|^2 + |mu|^2 - 2 x mu^T
        d2 = (
            (x * x).sum(dim=-1, keepdim=True)
            + (self.mu * self.mu).sum(dim=-1).unsqueeze(0)
            - 2.0 * (x @ self.mu.t())
        )
        env = torch.exp(-0.5 * torch.clamp(self.gamma, min=0.1).unsqueeze(0) * torch.clamp(d2, min=0.0))
        return env * torch.sin(self.linear(x))


class _FourierFilter(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, input_scale: float) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, hidden_dim)
        with torch.no_grad():
            self.linear.weight.normal_(0.0, input_scale / math.sqrt(in_dim))
            self.linear.bias.uniform_(-math.pi, math.pi)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.linear(x))


class _MFNNet(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_layers: int,
        input_scale: float,
        filter_type: str,
    ) -> None:
        super().__init__()
        flt_cls = _GaborFilter if filter_type == "gabor" else _FourierFilter
        self.filters = nn.ModuleList(
            [flt_cls(in_dim, hidden_dim, input_scale=input_scale) for _ in range(num_layers)]
        )
        self.linears = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers - 1)]
        )
        self.out = nn.Linear(hidden_dim, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.filters[0](x)
        for lin, flt in zip(self.linears, self.filters[1:]):
            z = lin(z) * flt(x)
        return self.out(z)


class MFNGridShiftModel(_BaseTorchINRModel):
    """Multiplicative Filter Network (GaborNet or FourierNet) 2D shift regressor."""

    def __init__(
        self,
        filter_type: str = "gabor",
        hidden_dim: int = 128,
        num_layers: int = 4,
        input_scale: float = 12.0,
        use_cyclic_cell_pos: bool = True,
        cyclic_cell_harmonics: int = 2,
        batch_size: int = 1024,
        lr: float = 2e-3,
        weight_decay: float = 1e-6,
        max_epochs: int = 150,
        patience: int = 25,
        val_fraction: float = 0.0,
        loss_type: str = "huber",
        device: str | None = None,
        random_state: int = 42,
    ) -> None:
        super().__init__()
        f_norm = filter_type.lower()
        if f_norm not in ("gabor", "fourier"):
            raise ValueError(f"Unsupported filter_type='{filter_type}'. Expected 'gabor' or 'fourier'.")
        self.filter_type = f_norm
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.input_scale = float(input_scale)
        self.use_cyclic_cell_pos = bool(use_cyclic_cell_pos)
        self.cyclic_cell_harmonics = int(cyclic_cell_harmonics)
        self.batch_size = int(batch_size)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.max_epochs = int(max_epochs)
        self.patience = int(patience)
        self.val_fraction = float(val_fraction)
        self.loss_type = str(loss_type).lower()
        self.random_state = int(random_state)
        self.device = self._init_device(device)

        self.encoder: SpatialFeatureEncoder | None = None
        self.net: nn.Module | None = None

    def _fit_impl(self, coords: np.ndarray, shifts: np.ndarray) -> None:
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        self.encoder = SpatialFeatureEncoder(
            normalize_inputs=True,
            use_fourier_features=False,
            use_cyclic_cell_pos=self.use_cyclic_cell_pos,
            cyclic_cell_harmonics=self.cyclic_cell_harmonics,
            random_state=self.random_state,
        )
        feats = self.encoder.fit_transform(coords).astype(np.float32)
        targets = shifts.astype(np.float32)
        self.net = _MFNNet(
            in_dim=feats.shape[1],
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            input_scale=self.input_scale,
            filter_type=self.filter_type,
        ).to(self.device)
        train_torch_regressor(
            net=self.net,
            forward_fn=self.net,
            inputs=(feats,),
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
        if self.encoder is None:
            raise RuntimeError("Model is not fitted yet.")
        feats = self.encoder.transform(coords).astype(np.float32)
        return self._predict_chunked(feats)


# ---------------------------------------------------------------------------
# V2-4: Spatial Mixture-of-Experts (MoE) & Overlapping Subdomain MLPs (FBPINN)
# ---------------------------------------------------------------------------


class _BatchedExpertsNet(nn.Module):
    """Vectorized K-expert 2-layer SiLU MLP + spatial gating or FBPINN windowing."""

    def __init__(
        self,
        in_dim: int,
        num_experts: int,
        hidden_dim: int,
        mode: str,
        fbpinn_grid: tuple[int, int],
    ) -> None:
        super().__init__()
        self.mode = mode
        self.num_experts = num_experts
        self.w1 = nn.Parameter(torch.randn(num_experts, in_dim, hidden_dim) / math.sqrt(in_dim))
        self.b1 = nn.Parameter(torch.zeros(num_experts, hidden_dim))
        self.w2 = nn.Parameter(torch.randn(num_experts, hidden_dim, hidden_dim) / math.sqrt(hidden_dim))
        self.b2 = nn.Parameter(torch.zeros(num_experts, hidden_dim))
        self.w3 = nn.Parameter(torch.randn(num_experts, hidden_dim, 2) / math.sqrt(hidden_dim))
        self.b3 = nn.Parameter(torch.zeros(num_experts, 2))
        self.act = nn.SiLU()

        if mode == "moe":
            self.gate = nn.Sequential(
                nn.Linear(2, 32),
                nn.SiLU(),
                nn.Linear(32, num_experts),
            )
        else:
            nx, ny = fbpinn_grid
            cx = torch.linspace(-1.0, 1.0, nx)
            cy = torch.linspace(-1.0, 1.0, ny)
            gx, gy = torch.meshgrid(cx, cy, indexing="ij")
            centers = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=-1)
            half_w = torch.tensor(
                [
                    1.6 / max(nx - 1, 1) if nx > 1 else 2.0,
                    1.6 / max(ny - 1, 1) if ny > 1 else 2.0,
                ],
                dtype=torch.float32,
            )
            self.register_buffer("centers", centers)
            self.register_buffer("half_w", half_w)

    def forward(self, feats: torch.Tensor, norm_xy: torch.Tensor) -> torch.Tensor:
        h = self.act(torch.einsum("bi,eih->beh", feats, self.w1) + self.b1.unsqueeze(0))
        h = self.act(torch.einsum("beh,ehk->bek", h, self.w2) + self.b2.unsqueeze(0))
        out = torch.einsum("bek,ekd->bed", h, self.w3) + self.b3.unsqueeze(0)

        if self.mode == "moe":
            weights = torch.softmax(self.gate(norm_xy), dim=-1)
        else:
            diff = (norm_xy.unsqueeze(1) - self.centers.unsqueeze(0)) / self.half_w.unsqueeze(0).unsqueeze(0)
            # Smooth raised-cosine window per axis
            w_axis = torch.where(
                diff.abs() < 1.0,
                0.5 * (1.0 + torch.cos(math.pi * diff)),
                torch.zeros_like(diff),
            )
            w_raw = w_axis[:, :, 0] * w_axis[:, :, 1] + 1e-6
            weights = w_raw / w_raw.sum(dim=-1, keepdim=True)

        return torch.einsum("be,bed->bd", weights, out)


class SpatialMoEGridShiftModel(_BaseTorchINRModel):
    """Spatial Mixture-of-Experts (MoE) or FBPINN overlapping subdomain MLPs."""

    def __init__(
        self,
        mode: str = "moe",
        num_experts: int = 8,
        fbpinn_grid: tuple[int, int] = (4, 4),
        hidden_dim: int = 64,
        use_fourier_features: bool = True,
        fourier_scales: tuple[float, ...] = (1.0, 3.0, 10.0),
        num_fourier_frequencies: int = 48,
        use_cyclic_cell_pos: bool = True,
        cyclic_cell_harmonics: int = 2,
        batch_size: int = 1024,
        lr: float = 2e-3,
        weight_decay: float = 1e-5,
        max_epochs: int = 150,
        patience: int = 25,
        val_fraction: float = 0.0,
        loss_type: str = "huber",
        device: str | None = None,
        random_state: int = 42,
    ) -> None:
        super().__init__()
        m_norm = mode.lower()
        if m_norm not in ("moe", "fbpinn"):
            raise ValueError(f"Unsupported mode='{mode}'. Expected 'moe' or 'fbpinn'.")
        self.mode = m_norm
        self.fbpinn_grid = (int(fbpinn_grid[0]), int(fbpinn_grid[1]))
        self.num_experts = (
            self.fbpinn_grid[0] * self.fbpinn_grid[1] if m_norm == "fbpinn" else int(num_experts)
        )
        self.hidden_dim = int(hidden_dim)
        self.use_fourier_features = bool(use_fourier_features)
        self.fourier_scales = tuple(float(s) for s in fourier_scales)
        self.num_fourier_frequencies = int(num_fourier_frequencies)
        self.use_cyclic_cell_pos = bool(use_cyclic_cell_pos)
        self.cyclic_cell_harmonics = int(cyclic_cell_harmonics)
        self.batch_size = int(batch_size)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.max_epochs = int(max_epochs)
        self.patience = int(patience)
        self.val_fraction = float(val_fraction)
        self.loss_type = str(loss_type).lower()
        self.random_state = int(random_state)
        self.device = self._init_device(device)

        self.encoder: SpatialFeatureEncoder | None = None
        self.net: _BatchedExpertsNet | None = None

    def _encode_pair(self, coords: np.ndarray, fit: bool) -> tuple[np.ndarray, np.ndarray]:
        assert self.encoder is not None
        feats = (self.encoder.fit_transform(coords) if fit else self.encoder.transform(coords)).astype(np.float32)
        norm_xy = feats[:, :2].copy()
        return feats, norm_xy

    def _fit_impl(self, coords: np.ndarray, shifts: np.ndarray) -> None:
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        self.encoder = SpatialFeatureEncoder(
            normalize_inputs=True,
            use_fourier_features=self.use_fourier_features,
            fourier_scales=self.fourier_scales,
            num_fourier_frequencies=self.num_fourier_frequencies,
            use_cyclic_cell_pos=self.use_cyclic_cell_pos,
            cyclic_cell_harmonics=self.cyclic_cell_harmonics,
            random_state=self.random_state,
        )
        feats, norm_xy = self._encode_pair(coords, fit=True)
        targets = shifts.astype(np.float32)
        self.net = _BatchedExpertsNet(
            in_dim=feats.shape[1],
            num_experts=self.num_experts,
            hidden_dim=self.hidden_dim,
            mode=self.mode,
            fbpinn_grid=self.fbpinn_grid,
        ).to(self.device)
        train_torch_regressor(
            net=self.net,
            forward_fn=self.net,
            inputs=(feats, norm_xy),
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
        if self.encoder is None or self.net is None:
            raise RuntimeError("Model is not fitted yet.")
        feats, norm_xy = self._encode_pair(coords, fit=False)
        self.net.eval()
        preds: list[np.ndarray] = []
        chunk_size = 16384
        with torch.no_grad():
            for i in range(0, len(feats), chunk_size):
                bf = torch.from_numpy(feats[i : i + chunk_size]).to(self.device)
                bn = torch.from_numpy(norm_xy[i : i + chunk_size]).to(self.device)
                preds.append(self.net(bf, bn).cpu().numpy())
        return np.vstack(preds).astype(np.float64)
