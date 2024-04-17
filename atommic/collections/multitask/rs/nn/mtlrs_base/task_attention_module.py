# coding=utf-8
__author__ = "Tim Paquaij"


#This code is based on the Task Attetniom Module described in https://openaccess.thecvf.com/content_ECCV_2018/papers/Zhenyu_Zhang_Joint_Task-Recursive_Learning_ECCV_2018_paper.pdf
# Residual Block was copied and adapted from: https://github.com/tengshaofeng/ResidualAttentionNetwork-pytorch/blob/master/Residual-Attention-Network/model/basic_layers.py
import torch
import torch.nn as nn

class TaskAttentionalModule(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.balance_conv1 = nn.Conv2d(int(in_channels * 2), int(in_channels), kernel_size=1)
        self.balance_conv2 = nn.Conv2d(int(in_channels), int(in_channels), kernel_size=1)
        self.residual_block =ResidualBlock(int(in_channels),int(in_channels),stride=1)
        self.fc = nn.Conv2d(int(in_channels * 2), in_channels, kernel_size=1)

    def forward(self, reconstruction_features, segmentation_features):
        # Balance unit
        concat_features = torch.cat((reconstruction_features, segmentation_features), dim=1)
        balance_tensor = torch.sigmoid(self.balance_conv1(concat_features))
        balanced_output = self.balance_conv2(balance_tensor * reconstruction_features + (1 - balance_tensor) * segmentation_features)
        # Conv-deconvolution layers for spatial attention
        res_block = torch.sigmoid(self.residual_block(balanced_output))
        # Generate gated features
        gated_depth_features = (1 + res_block) * reconstruction_features
        gated_segmentation_features = (1 + res_block) * segmentation_features

        # Concatenate and apply convolutional layer
        concatenated_features = torch.cat((gated_depth_features, gated_segmentation_features), dim=1)
        output = self.fc(concatenated_features)
        return output


class ResidualBlock(nn.Module):
    def __init__(self, input_channels, output_channels,stride =1):
        super().__init__()
        self.stride = stride

        self.input_channels = input_channels
        self.output_channels = output_channels
        self.res_block =nn.Sequential(nn.BatchNorm2d(input_channels),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(input_channels, int(output_channels/4),kernel_size= 1),
                        nn.BatchNorm2d(int(output_channels / 4)),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(int(output_channels / 4), int(output_channels / 8), kernel_size= 1),
                        nn.BatchNorm2d(int(output_channels / 8)),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(int(output_channels / 8), int(output_channels/4), kernel_size= 1),
                        nn.BatchNorm2d(int(output_channels / 4)),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(int(output_channels / 4), int(output_channels), kernel_size= 1))


    def forward(self, x):
        out = self.res_block(x)

        return out