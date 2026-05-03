from __future__ import annotations

import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def _module_tree(filename: str) -> ast.Module:
    return ast.parse((ROOT / filename).read_text(encoding="utf-8"))


class GridSetupStaticTests(unittest.TestCase):
    def test_grid_runner_imports_clean_question_columns(self) -> None:
        tree = _module_tree("run_grid.py")

        imported_names: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "phase0b_oracle":
                imported_names.update(alias.asname or alias.name for alias in node.names)

        self.assertIn("clean_question_columns", imported_names)

    def test_oracle_exposes_clean_question_columns(self) -> None:
        tree = _module_tree("phase0b_oracle.py")
        defined_functions = {
            node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
        }

        self.assertIn("clean_question_columns", defined_functions)


if __name__ == "__main__":
    unittest.main()
