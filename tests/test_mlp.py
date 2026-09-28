from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd
import torch
from src.data_utils import CSVDataset, batch_size_without_singleton
from src.model import MLP, predict_from_checkpoint


class MLPTests(unittest.TestCase):
    def test_invalid_labels_and_features_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.csv"
            for feature, label in [(1.0, 0.5), (1.0, 2), (np.nan, 1), (np.inf, 0)]:
                pd.DataFrame({"f1": [feature], "label": [label]}).to_csv(
                    path, index=False
                )
                with (
                    self.subTest(feature=feature, label=label),
                    self.assertRaises(ValueError),
                ):
                    CSVDataset(path, ["f1"], "label")

    def test_batchnorm_batches_include_every_row_without_singletons(self):
        for n_rows in [2, 31, 32, 33, 65, 97]:
            size = batch_size_without_singleton(n_rows)
            self.assertNotEqual(n_rows % size, 1)
            self.assertGreaterEqual(size, 2)
            self.assertLessEqual(size, n_rows)

    def test_checkpoint_keeps_scaler_and_feature_order(self):
        torch.manual_seed(42)
        model = MLP(2).eval()
        frame = pd.DataFrame({"second": [6.0, 9.0], "first": [3.0, 7.0]})
        standardized = (frame[["first", "second"]].to_numpy() - [1.0, 3.0]) / [2.0, 3.0]
        with torch.no_grad():
            expected = (
                model(torch.tensor(standardized, dtype=torch.float32))
                .softmax(dim=1)
                .numpy()
            )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "feature_columns": ["first", "second"],
                    "hidden": [64, 32],
                    "dropout": 0.2,
                    "scaler_mean": [1.0, 3.0],
                    "scaler_scale": [2.0, 3.0],
                },
                path,
            )
            np.testing.assert_allclose(
                predict_from_checkpoint(path, frame), expected, atol=1e-7
            )
            with self.assertRaises(ValueError):
                predict_from_checkpoint(path, frame.drop(columns="first"))


if __name__ == "__main__":
    unittest.main()
