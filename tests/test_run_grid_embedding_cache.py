import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import run_grid


class EmbeddingCacheTests(unittest.TestCase):
    def test_saved_cache_loads_without_pickle(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(run_grid, "OUTPUTS_DIR", Path(tmp)):
                with patch("phase0b_mapping.build_embeddings", return_value=np.array([[1.0], [2.0]])):
                    embeddings, var_codes = run_grid.load_or_build_survey_embeddings(
                        {"A": "alpha", "B": "beta"},
                        "unit",
                    )

                self.assertEqual(var_codes, ["A", "B"])
                np.testing.assert_array_equal(embeddings, np.array([[1.0], [2.0]]))

                cache_path = run_grid.survey_emb_cache_path("unit")
                with np.load(cache_path, allow_pickle=False) as cached:
                    self.assertNotEqual(cached["var_codes"].dtype, object)
                    self.assertEqual([str(code) for code in cached["var_codes"]], ["A", "B"])

                with patch("phase0b_mapping.build_embeddings") as build_embeddings:
                    cached_embeddings, cached_codes = run_grid.load_or_build_survey_embeddings(
                        {"A": "alpha", "B": "beta"},
                        "unit",
                    )

                build_embeddings.assert_not_called()
                self.assertEqual(cached_codes, ["A", "B"])
                np.testing.assert_array_equal(cached_embeddings, np.array([[1.0], [2.0]]))

    def test_legacy_object_cache_is_recomputed_without_pickle(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(run_grid, "OUTPUTS_DIR", Path(tmp)):
                cache_path = run_grid.survey_emb_cache_path("unit")
                np.savez(
                    cache_path,
                    embeddings=np.array([[99.0], [100.0]]),
                    var_codes=np.array(["A", "B"], dtype=object),
                )

                with patch("phase0b_mapping.build_embeddings", return_value=np.array([[3.0], [4.0]])):
                    embeddings, var_codes = run_grid.load_or_build_survey_embeddings(
                        {"A": "alpha", "B": "beta"},
                        "unit",
                    )

                self.assertEqual(var_codes, ["A", "B"])
                np.testing.assert_array_equal(embeddings, np.array([[3.0], [4.0]]))
                with np.load(cache_path, allow_pickle=False) as cached:
                    self.assertNotEqual(cached["var_codes"].dtype, object)
                    self.assertEqual([str(code) for code in cached["var_codes"]], ["A", "B"])


if __name__ == "__main__":
    unittest.main()
