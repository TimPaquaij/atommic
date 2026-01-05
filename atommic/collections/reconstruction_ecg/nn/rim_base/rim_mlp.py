# coding=utf-8
__author__ = "Tim Paquaij"

from typing import Any, Optional, Tuple, Union

import torch

import torch.nn as nn
import torch.nn.functional as F


class ProjectorMLP(nn.Module):
    def __init__(self, in_channels=64, hidden=512, out_dim=128, conv_dim=1):
        super().__init__()
        self.conv_dim = conv_dim
        if conv_dim == 1:
            self.pool = nn.AdaptiveAvgPool1d(1)  # [batch,92,1]
        elif conv_dim == 2:
            self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.mlp = nn.Sequential(nn.Linear(in_channels, hidden), nn.ReLU(), nn.Linear(hidden, out_dim))

    def forward(self, x):
        # x: [B, C, T] or [B, C, H, W]
        x = self.pool(x)
        x = x.flatten(1)  # ALWAYS [B, C]
        x = self.mlp(x)
        x = F.normalize(x, dim=-1)
        return x
