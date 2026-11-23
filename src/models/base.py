"""Abstract base classes for 2D grid shift regressors and coordinate transformers."""

from __future__ import annotations

import abc
import functools
import hashlib
import importlib.metadata
import inspect
import json
import logging
import pickle
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from src.data.catalog import GridDataset
    from src.evaluation.evaluator import EvaluationResult
    from src.viz.registry import PlotSpec, PlotType

_VOLATILE_CONFIG_KEYS: frozenset[str] = frozenset({"device", "spg_path"})


def _to_canonical_config_val(val: Any) -> Any:
    """Convert constructor parameter values into deterministic JSON-serializable structures."""
    if val is None or isinstance(val, (bool, int, float, str)):
        return val
    if isinstance(val, Enum):
        return val.value
    if isinstance(val, Path):
        return val.as_posix()
    if isinstance(val, (np.integer, np.floating, np.bool_)):
        return val.item()
    if isinstance(val, BaseSpatialModel):
        return {
            "model_class": f"{type(val).__module__}.{type(val).__qualname__}",
            "code_sha256": val._source_code_sha256(),
            "config": val.get_config_dict(),
        }
    if isinstance(val, (list, tuple)):
        return [_to_canonical_config_val(v) for v in val]
    if isinstance(val, dict):
        return {str(k): _to_canonical_config_val(v) for k, v in sorted(val.items())}
    if isinstance(val, pd.DataFrame):
        return f"<DataFrame shape={val.shape}>"
    if isinstance(val, np.ndarray):
        return f"<ndarray shape={val.shape} dtype={val.dtype}>"
    return repr(val)


@functools.lru_cache(maxsize=64)
def _file_sha256(file_path: str) -> str:
    """Return SHA-256 hex digest of a Python source file."""
    return hashlib.sha256(Path(file_path).read_bytes()).hexdigest()


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

    def get_config_dict(self) -> dict[str, Any]:
        """Inspect ``__init__`` parameters on ``self`` for deterministic fingerprinting."""
        cls = type(self)
        sig = inspect.signature(cls.__init__)
        params: dict[str, Any] = {}
        for param_name, param in sig.parameters.items():
            if param_name == "self" or param_name in _VOLATILE_CONFIG_KEYS:
                continue
            if param.kind in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            ):
                continue
            if hasattr(self, param_name):
                raw_val = getattr(self, param_name)
            elif hasattr(self, f"_{param_name}"):
                raw_val = getattr(self, f"_{param_name}")
            elif param.default is not inspect.Parameter.empty:
                raw_val = param.default
            else:
                raw_val = "<unset>"
            params[param_name] = _to_canonical_config_val(raw_val)
        return params

    def _source_code_sha256(self) -> str:
        """Compute combined SHA-256 of this model's source file and any nested child models."""
        hashes: list[str] = []
        try:
            src_file = inspect.getsourcefile(type(self))
            if src_file and Path(src_file).is_file():
                hashes.append(_file_sha256(str(Path(src_file).resolve())))
        except (TypeError, OSError):
            pass
        for attr_name in sorted(self.__dict__):
            val = self.__dict__[attr_name]
            if isinstance(val, BaseSpatialModel):
                hashes.append(f"{attr_name}:{val._source_code_sha256()}")
        return hashlib.sha256("|".join(hashes).encode("utf-8")).hexdigest()

    def compute_sync_fingerprint(self, train_grid: str | None = None) -> str:
        """Compute deterministic SHA-256 fingerprint over code, hyperparameters, and ``train_grid``.

        :param train_grid: Canonical training grid identifier (e.g., ``'ro_grid_4x4'``).
        :type train_grid: str | None
        :return: 64-character hexadecimal SHA-256 digest.
        :rtype: str
        """
        payload = {
            "model_class": f"{type(self).__module__}.{type(self).__qualname__}",
            "code_sha256": self._source_code_sha256(),
            "config": self.get_config_dict(),
            "train_grid": train_grid,
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

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

    def save_checkpoint(self, path: Path, train_grid: str | None = None) -> Path:
        """Persist this fitted model and its sync fingerprint as a ``.pkl`` envelope.

        :param path: Target ``.pkl`` file path.
        :type path: Path
        :param train_grid: Canonical training grid identifier used during fitting.
        :type train_grid: str | None
        :return: Written checkpoint path.
        :rtype: Path
        :raises RuntimeError: If called before the model is fitted.
        """
        if not self._is_fitted:
            raise RuntimeError(
                f"Cannot save unfitted {type(self).__name__} checkpoint."
            )
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        envelope = {
            "schema": "ml_transdat_model_v1",
            "fingerprint": self.compute_sync_fingerprint(train_grid),
            "train_grid": train_grid,
            "fit_time_sec": self._fit_time_sec,
            "model": self,
        }
        tmp_path = path.with_suffix(".pkl.tmp")
        with tmp_path.open("wb") as fh:
            pickle.dump(envelope, fh, protocol=pickle.HIGHEST_PROTOCOL)
        tmp_path.replace(path)
        return path

    def load_or_fit(
        self,
        train_data: tuple[np.ndarray, np.ndarray]
        | Callable[[], tuple[np.ndarray, np.ndarray]],
        *,
        cache_path: Path | None = None,
        train_grid: str | None = None,
        force_retrain: bool = False,
        label: str | None = None,
    ) -> BaseGridShiftModel:
        """Load a fingerprint-matched checkpoint if valid, or fit from scratch and save.

        :param train_data: Either ``(coords, shifts)`` or a lazy callable returning them.
        :type train_data: tuple[np.ndarray, np.ndarray] | Callable[[], tuple[np.ndarray, np.ndarray]]
        :param cache_path: Optional ``.pkl`` checkpoint path.
        :type cache_path: Path | None
        :param train_grid: Canonical training grid identifier for fingerprint verification.
        :type train_grid: str | None
        :param force_retrain: If True, ignore any existing checkpoint and retrain.
        :type force_retrain: bool
        :param label: Optional display label for log output.
        :type label: str | None
        :return: Fitted self instance.
        :rtype: BaseGridShiftModel
        """
        tag = label or type(self).__name__
        expected_fp = self.compute_sync_fingerprint(train_grid)

        if cache_path is not None and not force_retrain and Path(cache_path).is_file():
            try:
                with Path(cache_path).open("rb") as fh:
                    envelope = pickle.load(fh)
                if (
                    isinstance(envelope, dict)
                    and envelope.get("schema") == "ml_transdat_model_v1"
                    and envelope.get("fingerprint") == expected_fp
                    and isinstance(envelope.get("model"), type(self))
                ):
                    loaded: BaseGridShiftModel = envelope["model"]
                    self.__dict__.clear()
                    self.__dict__.update(loaded.__dict__)
                    print(
                        f"[CACHE HIT] Loaded {tag} from {Path(cache_path).as_posix()} "
                        f"(saved fit_time={self._fit_time_sec:.3f}s)"
                    )
                    return self
                print(
                    f"[CACHE INVALIDATED] Fingerprint mismatch for {tag} "
                    f"(code or params changed); retraining..."
                )
            except (OSError, pickle.PickleError, EOFError, AttributeError, RuntimeError) as exc:
                logger.warning("Failed to load checkpoint %s: %s", cache_path, exc)
                print(f"[CACHE CORRUPT] Retraining {tag}: {exc}")

        coords, shifts = train_data() if callable(train_data) else train_data
        self.fit(coords, shifts)
        if cache_path is not None:
            self.save_checkpoint(Path(cache_path), train_grid=train_grid)
            print(f"[CACHE SAVED] {tag} -> {Path(cache_path).as_posix()}")
        return self

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

    @property
    def fit_time_sec(self) -> float:
        """Return wall-clock fitting time in seconds (0.0 for analytical baselines)."""
        return 0.0

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

    @property
    def fit_time_sec(self) -> float:
        """Return the wrapped shift model's fitting time in seconds."""
        return float(self.shift_model.fit_time_sec)

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
