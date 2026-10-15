import sys
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import probe


class ProbeValidationTests(unittest.TestCase):
    def setUp(self):
        self.y = np.tile([0, 0, 0, 1], 25)
        self.x = np.arange(100).reshape(-1, 1)

    def test_invalid_fractions_and_empty_seeds_rejected(self):
        for fractions in ((), (0,), (-0.2,), (1.1,), (np.nan,), (np.inf,)):
            with self.subTest(fractions=fractions), self.assertRaises(ValueError):
                probe.label_efficiency(self.x, self.y, fracs=fractions)
        with self.assertRaises(ValueError):
            probe.evaluate(self.x, self.y, seeds=())

    def test_close_fractions_have_distinct_records(self):
        class Fixture:
            def fit(self, x, y):
                pass
            def predict_proba(self, x):
                return np.tile([0.5, 0.5], (len(x), 1))
        with patch.object(probe, "_probes", side_effect=lambda seed: {"linear": Fixture()}):
            result = probe.label_efficiency(self.x, self.y, fracs=(0.101, 0.104), seeds=(0,))
        self.assertEqual(set(result), {"0.101", "0.104"})

    def test_unsupported_labels_and_misaligned_features_rejected(self):
        for labels in (np.full(100, 2), np.ones(100), self.y.reshape(-1, 1)):
            with self.assertRaises(ValueError):
                probe.evaluate(self.x, labels)
        with self.assertRaises(ValueError):
            probe.evaluate(self.x[:-1], self.y)
        with self.assertRaises(ValueError):
            probe.evaluate(np.full((100, 1), np.nan), self.y)


if __name__ == "__main__":
    unittest.main()
