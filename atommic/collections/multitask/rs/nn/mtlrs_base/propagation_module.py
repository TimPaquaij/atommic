# coding=utf-8
__author__ = "Tim Paquaij"

#This code was copied and adapted from: https://github.com/SimonVandenhende/Multi-Task-Learning-PyTorch/blob/master/models/mti_net.py
import torch
import torch.nn as nn
from torch.functional import F
class FeaturePropagationModule(nn.Module):
    """ Feature Propagation Module """

    def __init__(self, num_tasks, per_task_channels):
        super().__init__()
        # General
        self.N = num_tasks
        self.per_task_channels = per_task_channels
        self.shared_channels = int(self.N * per_task_channels)

        # Non-linear function f
        downsample = nn.Sequential(nn.Conv2d(self.shared_channels, self.shared_channels // 4, 1, bias=False),
                                   nn.BatchNorm2d(self.shared_channels // 4))
        self.non_linear = nn.Sequential(
            BasicBlock(self.shared_channels, self.shared_channels // 4, downsample=downsample),
            BasicBlock(self.shared_channels // 4, self.shared_channels // 4),
            nn.Conv2d(self.shared_channels // 4, self.shared_channels, 1))

        # Dimensionality reduction
        downsample = nn.Sequential(nn.Conv2d(self.shared_channels, self.per_task_channels, 1, bias=False),
                                   nn.BatchNorm2d(self.per_task_channels))
        self.dimensionality_reduction = BasicBlock(self.shared_channels, self.per_task_channels,
                                                   downsample=downsample)

        # SEBlock
        self.se_rec = SEBlock(self.per_task_channels)

    def forward(self, reconstruction_features, segmentation_features):
        # Get shared representation
        concat = torch.cat([reconstruction_features,segmentation_features], 1)
        B, C, H, W = concat.size()
        shared = self.non_linear(concat)
        mask = F.softmax(shared.view(B, C // self.N, self.N, H, W), dim=2)  # Per task attention mask
        shared = torch.mul(mask, concat.view(B, C // self.N, self.N, H, W)).view(B, -1, H, W)

        # Perform dimensionality reduction
        shared = self.dimensionality_reduction(shared)

        # Per task squeeze-and-excitation
        reconstruction_features = self.se_rec(shared) + reconstruction_features  #Only the reconstruction features will be given attention

        return reconstruction_features


class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = self.conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = self.conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def conv3x3(self,in_planes, out_planes, stride=1, groups=1, dilation=1):
        """3x3 convolution with padding"""
        return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                         padding=dilation, groups=groups, bias=False, dilation=dilation)
    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class SEBlock(nn.Module):
    """ Squeeze-and-excitation block """
    def __init__(self, channels, r=16):
        super(SEBlock, self).__init__()
        self.r = r
        self.squeeze = nn.Sequential(nn.Linear(channels, channels//self.r),
                                     nn.ReLU(),
                                     nn.Linear(channels//self.r, channels),
                                     nn.Sigmoid())

    def forward(self, x):
        B, C, H, W = x.size()
        squeeze = self.squeeze(torch.mean(x, dim=(2,3))).view(B,C,1,1)
        return torch.mul(x, squeeze)