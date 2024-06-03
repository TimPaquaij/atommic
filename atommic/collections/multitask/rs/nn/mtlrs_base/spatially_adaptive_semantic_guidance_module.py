import torch
import torch.nn as nn
from torch.functional import F


class SASG(nn.Module):

    def __init__(self, channels_rec,channels_seg):
        super().__init__()
        self.conv = nn.Conv2d(channels_rec, channels_rec, kernel_size=1)
        self.spade  = SPADE(segmentation_channels=channels_seg,reconstruction_channels=channels_rec
                            )
        self.act = nn.LeakyReLU(negative_slope=0.2)


    def forward(self,hidden_layers_features,segmentation_prob):
        hidden_layers_features_s = self.spade(hidden_layers_features,segmentation_prob)
        hidden_layers_features_s = self.conv(self.act(hidden_layers_features_s))
        hidden_layers_features_s = self.spade(hidden_layers_features_s, segmentation_prob)
        hidden_layers_features_s = self.conv(self.act(hidden_layers_features_s))
        hidden_layers_features_att = hidden_layers_features_s+hidden_layers_features



        return hidden_layers_features_att

class SPADE(nn.Module):

    def __init__(self, segmentation_channels,reconstruction_channels):
        super().__init__()
        self.conv_1 = nn.Conv2d(segmentation_channels, segmentation_channels, kernel_size=1)
        self.conv_2 = nn.Conv2d(segmentation_channels, reconstruction_channels, kernel_size=1)
        self.instance = nn.InstanceNorm2d(64)
        self.act = nn.LeakyReLU(negative_slope=0.2)

    def forward(self,hidden_layers_features,segmentation_prob):
        hidden_layers_features = self.instance(hidden_layers_features)
        segmentation_prob = self.act(self.conv_1(segmentation_prob))
        hidden_layers_features_refined = torch.mul(self.conv_2(segmentation_prob),hidden_layers_features) + self.conv_2(segmentation_prob)

        return hidden_layers_features_refined