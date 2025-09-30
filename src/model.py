import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, in_dim, hidden=[64, 32], p_drop=0.2):
        super().__init__()
        layers = []
        dim_prev = in_dim
        for h in hidden:
            layers += [nn.Linear(dim_prev, h),
                       nn.BatchNorm1d(h),
                       nn.ReLU(),
                       nn.Dropout(p_drop)]
            dim_prev = h
        layers += [nn.Linear(dim_prev, 2)]
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)
