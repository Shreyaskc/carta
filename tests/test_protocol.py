"""Synthetic checks for feature handling and evaluation protocol."""
import sys
import ast
import os
import subprocess
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from datasets import DatasetSpec
from adapter import split_roles
from baselines import build_raw_matrix, build_freq_matrix
import probe
from src.config import TrainingConfig
from src.data.temporal_preprocessor import TabularPreprocessor


class ProtocolTests(unittest.TestCase):
    def test_baseline_cli_threshold_matches_reported_roles(self):
        import run_baselines
        frame = pd.DataFrame({"amount": [1., 2., 3., 4.],
                              "category": ["a", "a", "b", "b"],
                              "merchant": ["w", "x", "y", "z"],
                              "label": [0, 1, 0, 1]})
        spec = DatasetSpec("synthetic", ["amount"], ["category", "merchant"], "label")
        captured = []
        def record_raw(df, specification, threshold):
            captured.append(threshold)
            return np.zeros((4, 3))
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(run_baselines.datasets, "load", return_value=(frame, spec)), \
                 patch.object(run_baselines.baselines, "build_raw_matrix", side_effect=record_raw), \
                 patch.object(run_baselines.baselines, "pca_reduce", side_effect=lambda x, k: x), \
                 patch.object(run_baselines.probe, "evaluate", return_value={"linear": {"auprc": 0.5}, "gbdt": {"auprc": 0.5, "auc": 0.5}}), \
                 patch.object(run_baselines.probe, "label_efficiency", return_value={}), \
                 patch.object(run_baselines, "RESULTS", directory), \
                 patch.object(sys, "argv", ["run_baselines.py", "--card-threshold", "2"]):
                run_baselines.main()
        self.assertEqual(captured, [2])

    def test_output_override_and_repository_local_data(self):
        import datasets
        root = Path(__file__).resolve().parents[1]
        self.assertEqual(Path(datasets._LOCAL_IEEE), root / ".data" / "ieee.parquet")
        with tempfile.TemporaryDirectory() as directory:
            env = dict(os.environ, GATER_RESULTS=directory)
            result = subprocess.run([sys.executable, "-c", "import run_baselines; print(run_baselines.RESULTS)"],
                                    cwd=root, env=env, check=True, capture_output=True, text=True)
            self.assertEqual(result.stdout.strip(), directory)

    def test_preprocessor_maps_unseen_identifier_to_zero(self):
        frame = pd.DataFrame({"amount": np.arange(30, dtype=float),
                              "category": np.tile(["a", "b"], 15),
                              "merchant": [f"m{i}" for i in range(30)]})
        preprocessor = TabularPreprocessor(["amount"], ["category"], ["merchant"])
        preprocessor.fit(frame)
        held_out = pd.DataFrame({"amount": [5.], "category": ["a"], "merchant": ["new"]})
        output = preprocessor.transform(held_out)
        self.assertEqual(output["embedding"].tolist(), [[0]])
        self.assertEqual(output["continuous"].shape, (1, 1))
        self.assertEqual(output["one_hot"].shape, (1, 2))

    def test_preprocessor_requires_fit(self):
        preprocessor = TabularPreprocessor([], [], [])
        with self.assertRaises(RuntimeError):
            preprocessor.transform(pd.DataFrame(index=[0]))

    def test_label_efficiency_fits_only_training_rows(self):
        y = np.tile([0, 0, 0, 1], 25)
        x = np.arange(len(y)).reshape(-1, 1)
        train, test = probe.split(y, seed=5)
        fitted = []

        class RecordingProbe:
            def fit(self, features, labels):
                fitted.append(set(features[:, 0]))
            def predict_proba(self, features):
                return np.tile([0.5, 0.5], (len(features), 1))

        with patch.object(probe, "_probes", side_effect=lambda seed: {"linear": RecordingProbe()}):
            curve = probe.label_efficiency(x, y, fracs=(0.1, 0.5), seeds=(5,))
        self.assertEqual(set(curve), {"0.10", "0.50"})
        self.assertEqual(len(fitted), 2)
        for rows in fitted:
            self.assertTrue(rows <= set(train))
            self.assertFalse(rows & set(test))

    def test_stratified_split_is_disjoint_and_repeatable(self):
        y = np.tile([0, 0, 0, 1], 25)
        train, test = probe.split(y, seed=7)
        train_again, test_again = probe.split(y, seed=7)
        np.testing.assert_array_equal(train, train_again)
        np.testing.assert_array_equal(test, test_again)
        self.assertFalse(set(train) & set(test))
        self.assertEqual(len(train) + len(test), len(y))
        self.assertEqual(len(test), 30)
        self.assertEqual(set(y[train]), {0, 1})
        self.assertEqual(set(y[test]), {0, 1})

    def test_raw_drops_high_cardinality_and_frequency_baseline_keeps_it(self):
        frame = pd.DataFrame({"amount": [1., 2., 3., 4.],
                              "category": ["a", "a", "b", "b"],
                              "merchant": ["w", "x", "y", "z"]})
        spec = DatasetSpec("synthetic", ["amount"], ["category", "merchant"], "label")
        roles = split_roles(frame, spec, card_threshold=2)
        self.assertEqual(roles.one_hot, ["category"])
        self.assertEqual(roles.embedding, ["merchant"])
        raw = build_raw_matrix(frame, spec, card_threshold=2)
        frequency = build_freq_matrix(frame, spec, card_threshold=2)
        self.assertEqual(raw.shape, (4, 3))
        self.assertEqual(frequency.shape, (4, 4))
        np.testing.assert_allclose(frequency[:, -1], np.log1p(0.25))

    def test_disabled_annealing_is_fixed_beta_not_zero_kl(self):
        # Isolate the scalar schedule from unavailable training dependencies.
        path = Path(__file__).resolve().parents[1] / "src/training.py"
        tree = ast.parse(path.read_text())
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                        and n.name == "calculate_annealed_kl_weight")
        namespace = {"TrainingConfig": TrainingConfig, "jnp": np}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
        schedule = namespace[function.name]
        config = TrainingConfig(use_cyclical_annealing=False, kl_weight_max=0.7)
        for step in (0, 25, 99):
            self.assertEqual(schedule(config, step, 100), 0.7)






if __name__ == "__main__":
    unittest.main()
