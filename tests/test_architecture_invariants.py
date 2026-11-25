"""Automated architectural boundary, portability, and contract tests."""

from __future__ import annotations

import ast
import json
import re
import unittest
from pathlib import Path

from src import models as models_pkg
from src.models import BaseGridShiftModel
from src.models import __all__ as MODEL_EXPORTS

REPO_ROOT = Path(__file__).resolve().parent.parent
ALLOWED_EXTRA_EXP_FILES = {
    "experiments/optimal_surface_regressor/screen_families.py",
    "experiments/hybrid_surface_featgrid/screen_subgrid.py",
}


class TestArchitectureInvariants(unittest.TestCase):
    """Enforce layering, experiment structure, portability, and model contracts."""

    def test_src_never_imports_experiments_or_benchmarks(self) -> None:
        violations: list[str] = []
        for py_file in sorted((REPO_ROOT / "src").rglob("*.py")):
            tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name.startswith(("experiments", "benchmarks")):
                            violations.append(f"{py_file.relative_to(REPO_ROOT)}: import {alias.name}")
                elif isinstance(node, ast.ImportFrom) and node.module:
                    if node.module.startswith(("experiments", "benchmarks")):
                        violations.append(
                            f"{py_file.relative_to(REPO_ROOT)}: from {node.module} import ..."
                        )
        self.assertEqual(violations, [], f"Layering violations in src/: {violations}")

    def test_experiments_have_run_unit_and_valid_v2_brief(self) -> None:
        exp_root = REPO_ROOT / "experiments"
        required_keys = {
            "schema_version",
            "slug",
            "experiment_type",
            "status",
            "runtime_s",
            "key_metrics",
            "artifacts",
            "benchmark_candidates",
        }
        for exp_dir in sorted(p for p in exp_root.iterdir() if p.is_dir() and not p.name.startswith(".")):
            if exp_dir.name == "__pycache__":
                continue
            self.assertTrue(
                (exp_dir / "run_unit.py").is_file(),
                f"Missing run_unit.py in {exp_dir.name}",
            )
            brief_path = exp_dir / "unit_brief.json"
            self.assertTrue(brief_path.is_file(), f"Missing unit_brief.json in {exp_dir.name}")
            payload = json.loads(brief_path.read_text(encoding="utf-8"))
            self.assertEqual(payload.get("schema_version"), "2.0", f"{exp_dir.name} brief not v2.0")
            self.assertTrue(
                required_keys.issubset(payload.keys()),
                f"{exp_dir.name} missing keys: {required_keys - set(payload.keys())}",
            )
            for py_file in exp_dir.rglob("*.py"):
                if "__pycache__" in py_file.parts or py_file.name == "run_unit.py":
                    continue
                rel = py_file.relative_to(REPO_ROOT).as_posix()
                self.assertIn(
                    rel,
                    ALLOWED_EXTRA_EXP_FILES,
                    f"Unexpected helper module outside src/: {rel}",
                )

    def test_no_hardcoded_absolute_paths(self) -> None:
        pattern = re.compile(r"[A-Za-z]:\\\\Users\\\\|[A-Za-z]:/Users/|D:\\\\ProiecteRealizate|D:/ProiecteRealizate")
        hits: list[str] = []
        for folder in ("src", "experiments", "benchmarks", "tests"):
            for py_file in sorted((REPO_ROOT / folder).rglob("*.py")):
                if py_file.name == "test_architecture_invariants.py":
                    continue
                txt = py_file.read_text(encoding="utf-8")
                if pattern.search(txt):
                    hits.append(py_file.relative_to(REPO_ROOT).as_posix())
        self.assertEqual(hits, [], f"Hardcoded absolute paths found in: {hits}")

    def test_no_legacy_transect_66km_references_in_code(self) -> None:
        for folder in ("src", "tests"):
            for py_file in sorted((REPO_ROOT / folder).rglob("*.py")):
                if py_file.name == "test_architecture_invariants.py":
                    continue
                txt = py_file.read_text(encoding="utf-8").lower()
                self.assertNotIn("transect_66km", txt, f"Legacy transect_66km in {py_file}")

    def test_exported_models_support_config_and_fingerprint(self) -> None:
        for name in MODEL_EXPORTS:
            obj = getattr(models_pkg, name)
            if (
                isinstance(obj, type)
                and issubclass(obj, BaseGridShiftModel)
                and obj is not BaseGridShiftModel
            ):
                if name == "CompositeResidualShiftModel":
                    from src.models import CellBilinearShiftModel, RBFSurfaceRegressor

                    inst = obj(CellBilinearShiftModel(), RBFSurfaceRegressor())
                else:
                    inst = obj()
                cfg = inst.get_config_dict()
                fp = inst.compute_sync_fingerprint("ro_grid_4x4")
                self.assertIsInstance(cfg, dict, f"{name}.get_config_dict() must return dict")
                self.assertEqual(len(fp), 64, f"{name}.compute_sync_fingerprint() must be SHA-256")


if __name__ == "__main__":
    unittest.main()
