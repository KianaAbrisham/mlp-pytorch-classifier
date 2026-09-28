"""Binary tabular MLP and inference from its saved preprocessing metadata."""

import numpy as np
import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, in_dim, hidden=(64, 32), p_drop=0.2):
        super().__init__()
        if in_dim < 1 or any(width < 1 for width in hidden) or not 0 <= p_drop < 1:
            raise ValueError("Use positive layer widths and dropout in [0, 1).")
        layers = []
        previous = in_dim
        for width in hidden:
            layers.extend(
                [
                    nn.Linear(previous, width),
                    nn.BatchNorm1d(width),
                    nn.ReLU(),
                    nn.Dropout(p_drop),
                ]
            )
            previous = width
        layers.append(nn.Linear(previous, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def predict_from_checkpoint(checkpoint_path, frame):
    """Predict in stored feature order; uses CPU and the training scaler."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    columns = checkpoint["feature_columns"]
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing features: {sorted(missing)}")
    values = frame[columns].to_numpy(dtype=np.float64)
    if len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("Prediction features must be nonempty and finite.")
    values = (values - np.array(checkpoint["scaler_mean"])) / np.array(
        checkpoint["scaler_scale"]
    )
    model = MLP(len(columns), tuple(checkpoint["hidden"]), checkpoint["dropout"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    with torch.no_grad():
        return model(torch.tensor(values, dtype=torch.float32)).softmax(dim=1).numpy()
