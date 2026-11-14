"""Abstract base class and lifecycle protocol for standard Research Units."""

from __future__ import annotations

import abc
import json
from pathlib import Path
from typing import Any


class ResearchUnit(abc.ABC):
    """Base interface for all self-contained research units in ``experiments/``."""

    def __init__(self, unit_dir: Path) -> None:
        self.unit_dir = Path(unit_dir)
        self.results_dir = self.unit_dir / "results"
        self.plots_dir = self.unit_dir / "plots"
        self.data_dir = self.unit_dir / "data"

        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)

    @property
    @abc.abstractmethod
    def unit_id(self) -> str:
        """Sequential unit identifier (e.g., '01', '02')."""
        pass

    @property
    @abc.abstractmethod
    def unit_name(self) -> str:
        """Descriptive slug (e.g., 'spg_grid_analysis')."""
        pass

    @abc.abstractmethod
    def run(self) -> dict[str, Any]:
        """Execute the unit workflow, produce artifacts, and return summary dict."""
        pass

    def save_brief(self, payload: dict[str, Any]) -> Path:
        """Save standard machine-readable summary brief for AI agent consumption."""
        brief_path = self.unit_dir / "unit_brief.json"
        brief_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return brief_path
