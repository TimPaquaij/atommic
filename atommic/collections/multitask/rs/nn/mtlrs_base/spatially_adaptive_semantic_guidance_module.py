import torch
import torch.nn as nn
from torch.functional import F


class SASG(nn.Module):

    def __init__(self, channels):
        super().__init__()
        self.shared_channels = channels
        self.conv = nn.Conv2d(self.shared_channels, self.shared_channels, kernel_size=3)
        self.spade  = SPADE(channels=self.shared_channels)
        self.act = nn.LeakyReLU(negative_slope=0.2)


    def forward(self,hidden_layers_features,segmentation_prob):
        hidden_layers_features_s = self.spade(hidden_layers_features,segmentation_prob)
        hidden_layers_features_s = self.conv(self.act(hidden_layers_features_s))
        hidden_layers_features_s = self.spade(hidden_layers_features_s, segmentation_prob)
        hidden_layers_features_s = self.conv(self.act(hidden_layers_features_s))
        hidden_layers_features_att = hidden_layers_features_s+hidden_layers_features



        return hidden_layers_features_att

class SPADE(nn.Module):

    def __init__(self, channels):
        super().__init__()
        self.shared_channels = channels
        self.conv = nn.Conv2d(self.shared_channels, self.shared_channels, kernel_size=3)
        self.instance = nn.InstanceNorm2d(self.shared_channels)
        self.act = nn.LeakyReLU(negative_slope=0.2)

    def forward(self,hidden_layers_features,segmentation_prob):
        hidden_layers_features = self.instance(hidden_layers_features)

        segmentation_prob = self.act(self.conv(segmentation_prob))
        hidden_layers_features_refined = torch.mul(self.conv(segmentation_prob),hidden_layers_features) + self.conv(hidden_layers_features)

        return hidden_layers_features_refined