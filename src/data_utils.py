"""Load explicitly selected numeric CSV features and binary labels."""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class CSVDataset(Dataset):
    def __init__(self, csv_path, feature_cols, label_col):
        frame = pd.read_csv(csv_path)
        if (
            not feature_cols
            or len(set(feature_cols)) != len(feature_cols)
            or label_col in feature_cols
        ):
            raise ValueError("Choose unique feature columns, excluding the target.")
        missing = set(feature_cols + [label_col]) - set(frame.columns)
        if missing:
            raise ValueError(f"Missing columns: {sorted(missing)}")
        features = (
            frame[feature_cols]
            .apply(pd.to_numeric, errors="raise")
            .to_numpy(dtype=np.float32)
        )
        labels = pd.to_numeric(frame[label_col], errors="raise").to_numpy()
        if len(frame) == 0 or not np.isfinite(features).all():
            raise ValueError(
                "Features must be nonempty and finite; handle missing values explicitly."
            )
        if not np.isin(labels, [0, 1]).all():
            raise ValueError("Labels must be exactly 0 or 1, with no missing values.")
        self.X = torch.from_numpy(features)
        self.y = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.X[index], self.y[index]


def batch_size_without_singleton(n_rows, requested=32):
    """Keep every training row while avoiding a one-row BatchNorm batch."""
    if n_rows < 2 or requested < 2:
        raise ValueError("BatchNorm training needs at least two rows per batch.")
    size = min(requested, n_rows)
    while n_rows % size == 1:
        size += 1
    return size
