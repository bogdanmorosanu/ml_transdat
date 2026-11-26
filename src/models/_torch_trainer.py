"""Shared PyTorch regression training loop with early stopping and L-BFGS polishing."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def charbonnier_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Smooth L1-like Charbonnier loss for robust spatial regression."""
    return torch.mean(torch.sqrt((pred - target) ** 2 + 1e-8) - 1e-4)


def build_torch_loss(loss_type: str) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Return a picklable PyTorch regression loss function by name.

    :param loss_type: Loss identifier (``'mse'``, ``'huber'``, or ``'charbonnier'``).
    :type loss_type: str
    :return: Callable ``(pred, target) -> scalar loss``.
    :rtype: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    :raises ValueError: If ``loss_type`` is unsupported.
    """
    kind = loss_type.lower()
    if kind == "mse":
        return nn.MSELoss()
    if kind == "huber":
        return nn.SmoothL1Loss(beta=0.005)
    if kind == "charbonnier":
        return charbonnier_loss
    raise ValueError(f"Unsupported loss_type '{loss_type}'.")


def train_torch_regressor(
    net: nn.Module,
    forward_fn: Callable[..., torch.Tensor],
    inputs: tuple[np.ndarray, ...],
    targets: np.ndarray,
    *,
    batch_size: int,
    lr: float,
    weight_decay: float,
    max_epochs: int,
    patience: int,
    lr_scheduler: str = "cosine",
    val_fraction: float = 0.1,
    loss_type: str = "mse",
    lbfgs_steps: int = 0,
    lbfgs_history_size: int = 50,
    device: str = "cpu",
    random_state: int = 42,
) -> None:
    """Train a PyTorch regression module with AdamW, early stopping, and optional L-BFGS polish.

    :param net: Target PyTorch module already placed on ``device``.
    :type net: nn.Module
    :param forward_fn: Callable accepting unpacked input tensors and returning predictions.
    :type forward_fn: Callable[..., torch.Tensor]
    :param inputs: Tuple of float32 input numpy arrays of shape ``(N, D_i)``.
    :type inputs: tuple[np.ndarray, ...]
    :param targets: Float32 target numpy array of shape ``(N, 2)``.
    :type targets: np.ndarray
    :param batch_size: Mini-batch size for AdamW training.
    :type batch_size: int
    :param lr: Initial AdamW learning rate.
    :type lr: float
    :param weight_decay: AdamW weight decay penalty.
    :type weight_decay: float
    :param max_epochs: Maximum training epochs.
    :type max_epochs: int
    :param patience: Early-stopping patience in epochs.
    :type patience: int
    :param lr_scheduler: Scheduler type (``'cosine'`` or ``'plateau'``).
    :type lr_scheduler: str
    :param val_fraction: Validation split fraction (``<= 0`` trains on 100% of samples).
    :type val_fraction: float
    :param loss_type: Loss name (``'mse'``, ``'huber'``, or ``'charbonnier'``).
    :type loss_type: str
    :param lbfgs_steps: Optional post-AdamW L-BFGS iterations (``0`` disables).
    :type lbfgs_steps: int
    :param lbfgs_history_size: L-BFGS history size.
    :type lbfgs_history_size: int
    :param device: PyTorch device string (``'cpu'`` or ``'cuda'``).
    :type device: str
    :param random_state: Reproducibility seed.
    :type random_state: int
    """
    n_samples = len(targets)
    if val_fraction > 0.0 and n_samples >= 10:
        perm = np.random.RandomState(random_state).permutation(n_samples)
        val_size = max(int(val_fraction * n_samples), 1)
        val_idx = perm[:val_size]
        train_idx = perm[val_size:]
        val_tensors = tuple(torch.from_numpy(arr[val_idx]).to(device) for arr in inputs)
        val_y = torch.from_numpy(targets[val_idx]).to(device)
    else:
        train_idx = np.arange(n_samples)
        val_tensors = None
        val_y = None

    train_cpu_tensors = tuple(torch.from_numpy(arr[train_idx]) for arr in inputs)
    train_y_cpu = torch.from_numpy(targets[train_idx])

    train_gpu_tensors: tuple[torch.Tensor, ...] | None = None
    train_y_gpu: torch.Tensor | None = None
    loader: DataLoader[tuple[torch.Tensor, ...]] | None = None
    if torch.device(device).type == "cuda":
        try:
            train_gpu_tensors = tuple(t.to(device) for t in train_cpu_tensors)
            train_y_gpu = train_y_cpu.to(device)
        except torch.cuda.OutOfMemoryError:
            train_gpu_tensors = None
            train_y_gpu = None
            torch.cuda.empty_cache()

    if train_gpu_tensors is None:
        loader = DataLoader(
            TensorDataset(*train_cpu_tensors, train_y_cpu),
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
        )

    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
    if lr_scheduler.lower() == "cosine":
        scheduler: Any = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(max_epochs, 1), eta_min=1e-6
        )
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6
        )

    criterion = build_torch_loss(loss_type)
    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    no_improve = 0

    for _ in range(max_epochs):
        net.train()
        running_loss = torch.zeros((), device=device)
        n_batches = 0
        if train_gpu_tensors is not None and train_y_gpu is not None:
            n_train = train_y_gpu.shape[0]
            perm = torch.randperm(n_train, device=device)
            for start in range(0, n_train, batch_size):
                idx = perm[start : start + batch_size]
                bx = tuple(t[idx] for t in train_gpu_tensors)
                by = train_y_gpu[idx]
                optimizer.zero_grad(set_to_none=True)
                pred = forward_fn(*bx)
                loss = criterion(pred, by)
                loss.backward()
                optimizer.step()
                running_loss += loss.detach()
                n_batches += 1
        else:
            assert loader is not None
            for batch in loader:
                *bx, by = (t.to(device) for t in batch)
                optimizer.zero_grad(set_to_none=True)
                pred = forward_fn(*bx)
                loss = criterion(pred, by)
                loss.backward()
                optimizer.step()
                running_loss += loss.detach()
                n_batches += 1
        epoch_loss = float(running_loss.item())

        if val_tensors is not None and val_y is not None:
            net.eval()
            with torch.no_grad():
                mon_loss = float(criterion(forward_fn(*val_tensors), val_y).item())
        else:
            mon_loss = epoch_loss / max(n_batches, 1)

        if lr_scheduler.lower() == "cosine":
            scheduler.step()
        else:
            scheduler.step(mon_loss)

        if mon_loss < best_loss:
            best_loss = mon_loss
            best_state = {k: v.cpu().clone() for k, v in net.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            break

    if best_state is not None:
        net.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    if lbfgs_steps > 0:
        full_inputs = tuple(t.to(device) for t in train_cpu_tensors)
        full_y = train_y_cpu.to(device)
        net.eval()
        with torch.no_grad():
            pre_loss = float(criterion(forward_fn(*full_inputs), full_y).item())
        pre_state = {k: v.cpu().clone() for k, v in net.state_dict().items()}
        lbfgs = torch.optim.LBFGS(
            net.parameters(),
            lr=0.5,
            max_iter=lbfgs_steps,
            history_size=lbfgs_history_size,
            line_search_fn="strong_wolfe",
        )
        net.train()

        def _closure() -> torch.Tensor:
            lbfgs.zero_grad()
            out = forward_fn(*full_inputs)
            l_val = criterion(out, full_y)
            l_val.backward()
            return l_val

        lbfgs.step(_closure)
        net.eval()
        with torch.no_grad():
            post_loss = float(criterion(forward_fn(*full_inputs), full_y).item())
        if not np.isfinite(post_loss) or post_loss > pre_loss:
            net.load_state_dict({k: v.to(device) for k, v in pre_state.items()})

    net.eval()
