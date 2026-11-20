"""Abstract base classes for 2D grid shift regressors and coordinate transformers."""

from __future__ import annotations

import abc
import functools
import importlib.metadata
import inspect
import pickle
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from src.data.catalog import GridDataset
    from src.evaluation.evaluator import EvaluationResult
    from src.viz.registry import PlotSpec, PlotType


@functools.lru_cache(maxsize=64)
def _package_size_bytes(package_name: str) -> int:
    """Estimate installed disk footprint in bytes for a Python distribution."""
    try:
        dist_files = importlib.metadata.files(package_name)
    except importlib.metadata.PackageNotFoundError:
        return 0
    if not dist_files:
        return 0

    total = 0
    for f in dist_files:
        if f.size is not None:
            total += int(f.size)
        else:
            try:
                loc = Path(f.locate())
                if loc.is_file():
                    total += loc.stat().st_size
            except OSError:
                continue
    return total


@dataclass(frozen=True)
class MemoryBreakdown:
    """Four-component storage & memory breakdown for a spatial transformation model."""

    code_bytes: int = 0
    dependencies_bytes: int = 0
    model_instance_bytes: int = 0
    grid_lookup_bytes: int = 0

    @property
    def total_bytes(self) -> int:
        """Return total footprint across code, dependencies, instance, and lookup grid."""
        return (
            self.code_bytes
            + self.dependencies_bytes
            + self.model_instance_bytes
            + self.grid_lookup_bytes
        )

    @property
    def total_mb(self) -> float:
        """Return total footprint in megabytes (MB)."""
        return self.total_bytes / (1024.0 * 1024.0)

    def to_dict(self) -> dict[str, Any]:
        """Convert breakdown to a dictionary with byte counts and formatted MB values."""
        return {
            "code_bytes": self.code_bytes,
            "dependencies_bytes": self.dependencies_bytes,
            "model_instance_bytes": self.model_instance_bytes,
            "grid_lookup_bytes": self.grid_lookup_bytes,
            "total_bytes": self.total_bytes,
            "total_mb": self.total_mb,
        }


class BaseSpatialModel(abc.ABC):
    """Shared base class providing memory profiling and reporting hooks."""

    def _code_size_bytes(self) -> int:
        """Return disk size in bytes of the concrete model's source file."""
        try:
            mod_file = inspect.getfile(type(self))
            if mod_file and Path(mod_file).is_file():
                return Path(mod_file).stat().st_size
        except (TypeError, OSError):
            return 0
        return 0

    def _runtime_dependencies(self) -> tuple[str, ...]:
        """Return external PyPI distribution names required by this model at inference time."""
        return ()

    def _dependencies_bytes(self) -> int:
        """Return total installed byte footprint of declared runtime dependencies."""
        return sum(_package_size_bytes(pkg) for pkg in self._runtime_dependencies())

    def _grid_lookup_bytes(self) -> int:
        """Return byte size of retained spatial lookup anchors or cell tables."""
        return 0

    def memory_footprint_bytes(self) -> int:
        """Estimate serialized model size in bytes via pickle."""
        try:
            return len(pickle.dumps(self))
        except (TypeError, pickle.PicklingError):
            return 0

    def memory_breakdown(self) -> MemoryBreakdown:
        """Compute the 4-component memory breakdown (code, dependencies, model, grid)."""
        code_b = self._code_size_bytes()
        deps_b = self._dependencies_bytes()
        grid_lookup_b = self._grid_lookup_bytes()
        inst_b = max(0, self.memory_footprint_bytes() - grid_lookup_b)
        return MemoryBreakdown(
            code_bytes=code_b,
            dependencies_bytes=deps_b,
            model_instance_bytes=inst_b,
            grid_lookup_bytes=grid_lookup_b,
        )


class BaseGridShiftModel(BaseSpatialModel):
    """Abstract base class for 2D spatial distortion shift regressors.

    Uses the Template Method pattern: concrete :meth:`fit` and :meth:`predict`
    handle shape validation, fit timing, and fitted-state guards, while subclasses
    implement :meth:`_fit_impl` and :meth:`_predict_impl`.
    """

    def __init__(self) -> None:
        self._is_fitted: bool = False
        self._fit_time_sec: float = 0.0

    @property
    def is_fitted(self) -> bool:
        """Return True if the model has been fitted or initialized with lookup data."""
        return self._is_fitted

    @property
    def fit_time_sec(self) -> float:
        """Return wall-clock fitting time in seconds."""
        return self._fit_time_sec

    def memory_footprint_bytes(self) -> int:
        """Estimate serialized model size in bytes via pickle (0 if unfitted)."""
        if not self._is_fitted:
            return 0
        return super().memory_footprint_bytes()

    def fit(self, coords: np.ndarray, shifts: np.ndarray) -> BaseGridShiftModel:
        """Validate inputs, time execution, and fit the model on anchor data.

        :param coords: Anchor coordinates ``[st70_n, st70_e]`` in meters, shape ``(N, 2)``.
        :type coords: np.ndarray
        :param shifts: Target distortion shifts ``[dn, de]`` in meters, shape ``(N, 2)``.
        :type shifts: np.ndarray
        :return: Fitted self instance.
        :rtype: BaseGridShiftModel
        """
        c_arr, s_arr = self._validate_fit_inputs(coords, shifts)
        t0 = time.perf_counter()
        self._fit_impl(c_arr, s_arr)
        self._fit_time_sec = time.perf_counter() - t0
        self._is_fitted = True
        return self

    @abc.abstractmethod
    def _fit_impl(self, coords: np.ndarray, shifts: np.ndarray) -> None:
        """Algorithm-specific fitting logic on validated ``(N, 2)`` arrays."""

    def fit_stream(
        self, chunks: Iterator[tuple[np.ndarray, np.ndarray]]
    ) -> BaseGridShiftModel:
        """Fit the model from a generator yielding ``(coords_chunk, shifts_chunk)`` pairs.

        :param chunks: Iterator yielding ``(coords, shifts)`` batch pairs.
        :type chunks: Iterator[tuple[np.ndarray, np.ndarray]]
        :return: Fitted self instance.
        :rtype: BaseGridShiftModel
        :raises ValueError: If the chunk stream yields no samples.
        """
        coord_batches: list[np.ndarray] = []
        shift_batches: list[np.ndarray] = []
        for c_chunk, s_chunk in chunks:
            coord_batches.append(np.asarray(c_chunk, dtype=np.float64))
            shift_batches.append(np.asarray(s_chunk, dtype=np.float64))
        if not coord_batches:
            raise ValueError("Cannot fit from an empty chunk stream.")
        return self.fit(np.vstack(coord_batches), np.vstack(shift_batches))

    def predict(self, coords: np.ndarray) -> np.ndarray:
        """Validate fitted state and query coordinates, then predict 2D distortion shifts.

        :param coords: Query coordinates ``[st70_n, st70_e]`` in meters, shape ``(M, 2)``.
        :type coords: np.ndarray
        :return: Predicted shifts ``[pred_dn, pred_de]`` of shape ``(M, 2)``.
        :rtype: np.ndarray
        :raises RuntimeError: If called before the model is fitted.
        :raises ValueError: If ``coords`` does not have shape ``(M, 2)``.
        """
        if not self._is_fitted:
            raise RuntimeError(
                f"{type(self).__name__} must be fitted before calling predict()."
            )
        c_arr = self._validate_coords(coords, "coords")
        return self._predict_impl(c_arr)

    @abc.abstractmethod
    def _predict_impl(self, coords: np.ndarray) -> np.ndarray:
        """Algorithm-specific prediction logic on validated ``(M, 2)`` coordinates."""

    def assess(
        self,
        dataset_or_coords: GridDataset | np.ndarray,
        true_shifts: np.ndarray | None = None,
        model_name: str | None = None,
    ) -> EvaluationResult:
        """Evaluate this model's precision against a :class:`GridDataset` or raw arrays.

        :param dataset_or_coords: Either a :class:`GridDataset` or ``(N, 2)`` query coordinates.
        :type dataset_or_coords: GridDataset | np.ndarray
        :param true_shifts: Ground-truth ``(N, 2)`` shifts (required when passing raw coordinates).
        :type true_shifts: np.ndarray | None
        :param model_name: Optional display name (defaults to class name).
        :type model_name: str | None
        :return: Standardized :class:`EvaluationResult`.
        :rtype: EvaluationResult
        :raises ValueError: If raw coordinates are passed without ``true_shifts``.
        """
        from src.data.catalog import GridDataset, GridId
        from src.evaluation.evaluator import ModelEvaluator

        name = model_name or type(self).__name__
        evaluator = ModelEvaluator()

        if isinstance(dataset_or_coords, GridDataset):
            return evaluator.evaluate_shift_model(
                self, dataset_or_coords, model_name=name
            )

        if true_shifts is None:
            raise ValueError(
                "true_shifts must be provided when assessing with raw coordinate arrays."
            )
        c_arr = self._validate_coords(dataset_or_coords, "coords")
        s_arr = self._validate_coords(true_shifts, "true_shifts")
        if c_arr.shape[0] != s_arr.shape[0]:
            raise ValueError(
                f"Sample count mismatch: coords={c_arr.shape[0]}, true_shifts={s_arr.shape[0]}"
            )
        df_tmp = pd.DataFrame(
            {
                "st70_n": c_arr[:, 0],
                "st70_e": c_arr[:, 1],
                "shift_dn_m": s_arr[:, 0],
                "shift_de_m": s_arr[:, 1],
            }
        )
        ds_tmp = GridDataset(grid_id=GridId.CUSTOM, df=df_tmp)
        return evaluator.evaluate_shift_model(self, ds_tmp, model_name=name)

    def report(self, eval_result: EvaluationResult) -> str:
        """Format a human-readable evaluation scorecard for an :class:`EvaluationResult`."""
        from src.evaluation.metrics import format_scorecard_table

        return format_scorecard_table(
            eval_result.summary,
            eval_result.model_name,
            eval_result.dataset_id,
        )

    def plot(
        self,
        eval_result: EvaluationResult,
        out_dir: Path,
        plots: PlotSpec | None = None,
        dpi: int = 200,
        **kwargs: Any,
    ) -> dict[PlotType, Path]:
        """Render diagnostic plots for an :class:`EvaluationResult` using :class:`PlotDispatcher`."""
        from src.viz.dispatcher import PlotDispatcher
        from src.viz.registry import PlotGroup

        active_plots = plots if plots is not None else PlotGroup.STANDARD_DIAGNOSTIC
        return PlotDispatcher().render(
            eval_result=eval_result,
            out_dir=out_dir,
            plots=active_plots,
            dpi=dpi,
            **kwargs,
        )

    @staticmethod
    def _validate_coords(coords: np.ndarray, name: str = "coords") -> np.ndarray:
        arr = np.asarray(coords, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError(f"{name} must have shape (N, 2), got {arr.shape}")
        return arr

    @classmethod
    def _validate_fit_inputs(
        cls, coords: np.ndarray, shifts: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        c_arr = cls._validate_coords(coords, "coords")
        s_arr = cls._validate_coords(shifts, "shifts")
        if c_arr.shape[0] != s_arr.shape[0]:
            raise ValueError(
                f"Sample count mismatch: coords={c_arr.shape[0]}, shifts={s_arr.shape[0]}"
            )
        return c_arr, s_arr


class BaseCoordinateTransformer(BaseSpatialModel):
    """Abstract base class for full ETRS89 <-> Stereo 70 coordinate transformers."""

    @abc.abstractmethod
    def etrs89_to_st70(
        self,
        lat_rad: Sequence[float] | np.ndarray,
        lon_rad: Sequence[float] | np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Transform ETRS89 ``(lat_rad, lon_rad)`` to Stereo 70 ``(N, E)`` in meters."""

    @abc.abstractmethod
    def st70_to_etrs89(
        self,
        n: Sequence[float] | np.ndarray,
        e: Sequence[float] | np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Transform Stereo 70 ``(N, E)`` in meters to ETRS89 ``(lat_rad, lon_rad)``."""

    def assess(
        self,
        dataset: GridDataset,
        direction: str = "etrs89_to_st70",
        model_name: str | None = None,
    ) -> EvaluationResult:
        """Evaluate this transformer against a :class:`GridDataset`."""
        from src.evaluation.evaluator import ModelEvaluator

        name = model_name or type(self).__name__
        return ModelEvaluator().evaluate_transformer(
            transformer=self,
            dataset=dataset,
            direction=direction,
            model_name=name,
        )

    def report(self, eval_result: EvaluationResult) -> str:
        """Format a human-readable evaluation scorecard for an :class:`EvaluationResult`."""
        from src.evaluation.metrics import format_scorecard_table

        return format_scorecard_table(
            eval_result.summary,
            eval_result.model_name,
            eval_result.direction,
        )

    def plot(
        self,
        eval_result: EvaluationResult,
        out_dir: Path,
        plots: PlotSpec | None = None,
        dpi: int = 200,
        **kwargs: Any,
    ) -> dict[PlotType, Path]:
        """Render diagnostic plots for an :class:`EvaluationResult` using :class:`PlotDispatcher`."""
        from src.viz.dispatcher import PlotDispatcher
        from src.viz.registry import PlotGroup

        active_plots = plots if plots is not None else PlotGroup.STANDARD_DIAGNOSTIC
        return PlotDispatcher().render(
            eval_result=eval_result,
            out_dir=out_dir,
            plots=active_plots,
            dpi=dpi,
            **kwargs,
        )


class ResidualCoordinateTransformer(BaseCoordinateTransformer):
    """Composes ``StereoProj + Helmert2D + BaseGridShiftModel`` into a full transformer.

    Matches ``pytransdat.TransRO`` grid shift sign convention:
    - ``st70_to_etrs89`` subtracts predicted shifts before Helmert/inverse projection.
    - ``etrs89_to_st70`` adds predicted shifts after projection/inverse Helmert.
    """

    def __init__(
        self,
        base_transformer: BaseCoordinateTransformer,
        shift_model: BaseGridShiftModel,
    ) -> None:
        if not shift_model.is_fitted:
            raise RuntimeError(
                "shift_model must be fitted before wrapping in ResidualCoordinateTransformer."
            )
        self.base_transformer = base_transformer
        self.shift_model = shift_model

    def _runtime_dependencies(self) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                (
                    *self.base_transformer._runtime_dependencies(),
                    *self.shift_model._runtime_dependencies(),
                )
            )
        )

    def _grid_lookup_bytes(self) -> int:
        return self.shift_model._grid_lookup_bytes()

    def st70_to_etrs89(
        self,
        n: Sequence[float] | np.ndarray,
        e: Sequence[float] | np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        n_arr = np.asarray(n, dtype=np.float64).ravel()
        e_arr = np.asarray(e, dtype=np.float64).ravel()
        shifts = self.shift_model.predict(np.column_stack([n_arr, e_arr]))
        # Matches pytransdat TransRO.st70_to_etrs89: Grid2D.trans(n, e, -1)
        return self.base_transformer.st70_to_etrs89(
            n_arr - shifts[:, 0], e_arr - shifts[:, 1]
        )

    def etrs89_to_st70(
        self,
        lat_rad: Sequence[float] | np.ndarray,
        lon_rad: Sequence[float] | np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        n_approx, e_approx = self.base_transformer.etrs89_to_st70(lat_rad, lon_rad)
        # Matches pytransdat TransRO.etrs89_to_st70: Grid2D.trans(n_approx, e_approx, +1)
        shifts = self.shift_model.predict(np.column_stack([n_approx, e_approx]))
        return n_approx + shifts[:, 0], e_approx + shifts[:, 1]
