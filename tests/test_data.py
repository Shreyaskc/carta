"""Public-loader and category-vocabulary checks using only in-memory fixtures."""
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import pandas as pd
import numpy as np
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import datasets
from src.data.temporal_preprocessor import TabularPreprocessor


class DataTests(unittest.TestCase):
    def test_seeded_quantiles_and_serialization_are_repeatable(self):
        frame = pd.DataFrame({"amount": np.arange(20000, dtype=float) ** 2})
        first = TabularPreprocessor(["amount"], [], [], random_state=7).fit(frame)
        second = TabularPreprocessor(["amount"], [], [], random_state=7).fit(frame)
        np.testing.assert_array_equal(first.transform(frame)["continuous"], second.transform(frame)["continuous"])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "preprocessor.joblib"
            first.save(path)
            restored = TabularPreprocessor.load(path)
            np.testing.assert_array_equal(first.transform(frame)["continuous"], restored.transform(frame)["continuous"])

    def test_local_ulb_accepts_original_and_normalized_label_schema(self):
        original = pd.DataFrame({"V1": [1., 2.], "Class": ["0", "1"]})
        normalized = pd.DataFrame({"V1": [1., 2.], "is_fraud": [0, 1]})
        outputs = []
        for frame in (original, normalized):
            with patch.object(datasets.os.path, "exists", return_value=True), \
                 patch.object(datasets.pd, "read_parquet", return_value=frame):
                result, spec = datasets.load_creditcard()
            self.assertEqual(spec.numeric, ["V1"])
            outputs.append(result)
        pd.testing.assert_frame_equal(*outputs)

    def test_reserved_unknown_category_does_not_create_invalid_ids(self):
        preprocessor = TabularPreprocessor([], [], ["merchant"])
        preprocessor.fit(pd.DataFrame({"merchant": ["<unknown>", "a"]}))
        output = preprocessor.transform(pd.DataFrame({"merchant": ["<unknown>", "a", "new"]}))
        ids = output["embedding"][:, 0]
        self.assertEqual(ids.tolist(), [0, 1, 0])
        self.assertTrue((ids < preprocessor.metadata["embedding_cardinalities"][0]).all())

    def test_csv_and_parquet_head_limits_match(self):
        raw = pd.DataFrame({"Amount": [f"${i}" for i in range(5)], "Is Fraud?": ["No", "Yes", "No", "Yes", "No"],
                            "Time": ["12:00"] * 5, "Year": [2020] * 5, "Month": [1] * 5, "Day": [1] * 5,
                            "Use Chip": ["Swipe"] * 5, "MCC": [1] * 5, "Merchant State": ["CA"] * 5,
                            "Merchant Name": [str(i) for i in range(5)], "Merchant City": ["x"] * 5, "Zip": ["1"] * 5})
        outputs = []
        for extension in ("csv", "parquet"):
            with patch.dict(os.environ, {"TABFORMER_NROWS": "2", "TABFORMER_MINFREQ": "1"}, clear=True), \
                 patch.object(datasets, "_tabformer_file", return_value=f"fixture.{extension}"), \
                 patch.object(datasets.pd, "read_parquet", return_value=raw), \
                 patch.object(datasets.pd, "read_csv", side_effect=lambda path, nrows: raw.head(nrows)):
                frame, spec = datasets.load_tabformer()
            self.assertEqual(spec.n_rows, 2)
            outputs.append(frame)
        pd.testing.assert_frame_equal(*outputs)


if __name__ == "__main__":
    unittest.main()
