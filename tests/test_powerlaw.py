"""Test transform numerics with a deterministic detector stub, not detection quality."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd


class PowerlawTests(unittest.TestCase):
    def load_transformer(self, ratio=1):
        path = Path(__file__).resolve().parents[1] / "src/data/data_processing.py"
        detector = SimpleNamespace(Fit=lambda data, verbose: SimpleNamespace(distribution_compare=lambda *args, **kwargs: (ratio, 0.01)))
        specification = importlib.util.spec_from_file_location("powerlaw_numeric_test", path)
        module = importlib.util.module_from_spec(specification)
        with patch.dict(sys.modules, {"powerlaw": detector}):
            specification.loader.exec_module(module)
        return module.SmartPowerlawTransformer

    def test_refit_matches_fresh_transform(self):
        frame = pd.DataFrame({"x": [1., 2., 4., 8., 16.]})
        cls = self.load_transformer()
        transformer = cls(continuous_cols=["x"])
        once = transformer.fit_transform(frame)
        twice = transformer.fit_transform(frame)
        pd.testing.assert_frame_equal(once, twice)
        self.assertEqual(transformer.transformed_columns_, ["x"])

    def test_out_of_domain_and_nonfinite_values_rejected(self):
        transformer = self.load_transformer()(continuous_cols=["x"])
        transformer.fit(pd.DataFrame({"x": [1., 2., 4., 8., 16.]}))
        for value in (-2., 0., np.inf, np.nan):
            with self.subTest(value=value), self.assertRaises(ValueError):
                transformer.transform(pd.DataFrame({"x": [value]}))


if __name__ == "__main__":
    unittest.main()
