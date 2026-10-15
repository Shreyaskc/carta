"""CLI checks with encoding/subprocesses replaced by synthetic test fixtures."""
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import gater_core
import gater_runner
from datasets import DatasetSpec


class CliTests(unittest.TestCase):
    def test_encoder_output_accepts_bare_nested_and_absolute_paths(self):
        spec = DatasetSpec("synthetic", [], [], "label", n_rows=2, row_fingerprint="synthetic-fixture")
        with tempfile.TemporaryDirectory() as directory:
            cwd = os.getcwd()
            try:
                os.chdir(directory)
                for output in ("embedding.npy", "nested/embedding.npy", str(Path(directory) / "absolute.npy")):
                    with patch.object(gater_core, "train_and_encode", return_value=(np.ones((2, 16)), spec)), \
                         patch.object(sys, "argv", ["gater_runner.py", "--out", output]):
                        gater_runner.main()
                    self.assertEqual(np.load(output).shape, (2, 16))
            finally:
                os.chdir(cwd)

    def test_cluster_stability_rejects_insufficient_runs(self):
        import cluster_stability
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "embedding.npy"
            np.save(path, np.ones((4, 2)))
            for args in (("--runs", "1"), ("--k", "5")):
                with patch.object(sys, "argv", ["cluster_stability.py", "--emb", str(path), *args]):
                    with self.assertRaises(ValueError):
                        cluster_stability.main()

    def test_reference_script_import_does_not_execute(self):
        path = Path(__file__).resolve().parents[1] / "run_c2_reference.py"
        specification = importlib.util.spec_from_file_location("reference_import_test", path)
        module = importlib.util.module_from_spec(specification)
        cwd = os.getcwd()
        with patch("subprocess.run") as run:
            specification.loader.exec_module(module)
        run.assert_not_called()
        self.assertEqual(os.getcwd(), cwd)


if __name__ == "__main__":
    unittest.main()
