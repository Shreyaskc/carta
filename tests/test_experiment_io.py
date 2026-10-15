from pathlib import Path
import sys
import tempfile
import unittest
import json
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from datasets import DatasetSpec, describe_population
from experiment_io import save_embedding, load_embedding, require_matching_protocol, validate_reference_overrides, require_baseline_compatibility, dataset_fingerprint, evaluation_fingerprint


class ExperimentIoTests(unittest.TestCase):
    def test_reused_protocol_and_baseline_population_are_checked(self):
        frame = pd.DataFrame({"amount": [1., 2.], "label": [0, 1]})
        spec = describe_population(frame, DatasetSpec("synthetic", ["amount"], [], "label"))
        spec.encoding_protocol = {"encoder": "mlp", "epochs": 1}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "embedding.npy"
            save_embedding(path, np.ones((2, 16)), spec)
            with self.assertRaises(ValueError):
                load_embedding(path, frame, spec, expected_protocol={"encoder": "perceiver", "epochs": 999})
        baseline = {"dataset": "synthetic", "seeds": [0], "row_fingerprint": dataset_fingerprint(frame, spec), "evaluation_fingerprint": evaluation_fingerprint(frame, spec)}
        require_baseline_compatibility(baseline, frame, spec, [0])
        with self.assertRaises(ValueError):
            require_baseline_compatibility(baseline, frame.iloc[::-1], spec, [0])

    def test_partial_comparison_rejects_protocol_change_before_any_run(self):
        import run_representation_comparison as comparison
        frame = pd.DataFrame({"amount": [1., 2.], "label": [0, 1]})
        spec = DatasetSpec("synthetic", ["amount"], [], "label")
        with tempfile.TemporaryDirectory() as directory:
            record = {"dataset": "synthetic", "seeds": [9], "epochs": 5, "methods": {"pca16": {}}}
            (Path(directory) / "representation_comparison_synthetic.json").write_text(json.dumps(record))
            with patch.object(comparison, "RESULTS", directory), patch.object(comparison, "DS", "synthetic"), \
                 patch.object(comparison, "METHODS", ["pca16"]), patch.object(comparison.datasets, "load", return_value=(frame, spec)), \
                 patch.object(comparison, "embed") as encode, self.assertRaises(ValueError):
                comparison.main()
            encode.assert_not_called()

    def test_embedding_rejects_equal_count_reordered_rows(self):
        frame = pd.DataFrame({"amount": [1., 2., 3.], "label": [0, 1, 0]})
        spec = describe_population(frame, DatasetSpec("synthetic", ["amount"], [], "label"))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "embedding.npy"
            save_embedding(path, np.ones((3, 16)), spec)
            self.assertEqual(load_embedding(path, frame, spec).shape, (3, 16))
            with self.assertRaises(ValueError):
                load_embedding(path, frame.iloc[::-1], spec)

    def test_sample_metadata_and_protocol_merge(self):
        frame = pd.DataFrame({"amount": [1., 2.], "label": [0, 1]})
        spec = describe_population(frame, DatasetSpec("synthetic", ["amount"], [], "label", n_rows=100))
        self.assertEqual((spec.n_rows, spec.fraud_rate), (2, 0.5))
        old = {"dataset": "synthetic", "seeds": [0, 1], "epochs": 10, "methods": {"pca16": {}}, "evaluation_fingerprint": "population"}
        require_matching_protocol(old, "synthetic", [0, 1], 10, "population")
        with self.assertRaises(ValueError):
            require_matching_protocol(old, "synthetic", [2, 3], 20, "population")
        with self.assertRaises(ValueError):
            validate_reference_overrides({"random_state": 7})

    def test_content_replacement_and_invalid_save_fail(self):
        frame = pd.DataFrame({"amount": [1., 2.], "label": [0, 1]})
        spec = describe_population(frame, DatasetSpec("synthetic", ["amount"], [], "label"))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "embedding.npy"
            save_embedding(path, np.arange(4).reshape(2, 2), spec)
            np.save(path, np.arange(4).reshape(2, 2)[::-1])
            with self.assertRaisesRegex(ValueError, "checksum"):
                load_embedding(path, frame, spec)
            for array in (np.ones(2), np.ones((1, 2)), np.full((2, 2), np.nan)):
                with self.assertRaises(ValueError):
                    save_embedding(Path(directory) / "invalid.npy", array, spec)
            self.assertFalse((Path(directory) / "invalid.npy").exists())

    def test_merge_rejects_changed_features_and_labels(self):
        frame = pd.DataFrame({"amount": [1., 2.], "label": [0, 1]})
        spec = DatasetSpec("synthetic", ["amount"], [], "label")
        population = evaluation_fingerprint(frame, spec)
        old = {"dataset": "synthetic", "seeds": [0], "epochs": 10,
               "evaluation_fingerprint": population, "methods": {"pca16": {}}}
        require_matching_protocol(old, "synthetic", [0], 10, population)
        for changed in (frame.assign(amount=[4., 5.]), frame.assign(label=[1, 0])):
            with self.assertRaises(ValueError):
                require_matching_protocol(old, "synthetic", [0], 10, evaluation_fingerprint(changed, spec))
        with self.assertRaises(ValueError):
            require_matching_protocol({k: v for k, v in old.items() if k != "evaluation_fingerprint"},
                                      "synthetic", [0], 10, population)

    def test_full_reused_recipe_checks_nested_training(self):
        from run_representation_comparison import carta_protocol
        frame = pd.DataFrame({"amount": [1., 2.], "label": [0, 1]})
        spec = describe_population(frame, DatasetSpec("synthetic", ["amount"], [], "label"))
        spec.encoding_protocol = carta_protocol()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "embedding.npy"
            save_embedding(path, np.ones((2, 16)), spec)
            _, metadata = load_embedding(path, frame, spec, expected_protocol=carta_protocol(), return_metadata=True)
            self.assertEqual(metadata["verification"], "verified rows and content")
            spec.encoding_protocol["training_config"]["augmentation_mask_prob"] = 0.99
            save_embedding(path, np.ones((2, 16)), spec)
            with self.assertRaisesRegex(ValueError, "protocol"):
                load_embedding(path, frame, spec, expected_protocol=carta_protocol())

    def test_legacy_embedding_requires_explicit_opt_in(self):
        frame = pd.DataFrame({"amount": [1., 2.], "label": [0, 1]})
        spec = DatasetSpec("synthetic", ["amount"], [], "label")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.npy"
            np.save(path, np.ones((2, 16)))
            with self.assertRaises(ValueError):
                load_embedding(path, frame, spec)
            with self.assertWarns(RuntimeWarning):
                load_embedding(path, frame, spec, allow_unverified=True)


if __name__ == "__main__":
    unittest.main()
