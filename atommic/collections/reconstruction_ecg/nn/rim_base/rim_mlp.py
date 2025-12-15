# coding=utf-8
__author__ = "Tim Paquaij"

from typing import Any, Optional, Tuple, Union

import torch

import torch.nn as nn
import torch.nn.functional as F
class ProjectorMLP(nn.Module):
    def __init__(self, in_channels=64, hidden=512, out_dim=128):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)   # [batch,92,1]
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim)
        )

    def forward(self, x):
        # x: [batch, 92, 5000]
        x = self.pool(x).squeeze(-1)          # [batch, 92]
        x = F.normalize(self.mlp(x), dim=-1)  # [batch, out_dim]
        return x