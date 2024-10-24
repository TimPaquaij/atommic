# coding=utf-8
__author__ = "Tim Paquaij"


#This code is based on the Task Attetniom Module described in https://openaccess.thecvf.com/content_ECCV_2018/papers/Zhenyu_Zhang_Joint_Task-Recursive_Learning_ECCV_2018_paper.pdf
# Residual Block was copied and adapted from: https://github.com/tengshaofeng/ResidualAttentionNetwork-pytorch/blob/master/Residual-Attention-Network/model/basic_layers.py
import torch
import torch.nn as nn

class TaskAttentionalModule(nn.Module):
    def __init__(self, in_channels,kernel_size,padding):
        super().__init__()
        self.balance_conv1 = nn.Conv2d(int(in_channels * 2), int(in_channels), kernel_size=kernel_size,padding=padding)
        self.balance_conv2 = nn.Conv2d(int(in_channels), int(in_channels), kernel_size=kernel_size,padding=padding)
        self.residual_block =ResidualBlock(int(in_channels),int(in_channels))
        self.fc = nn.Conv2d(int(in_channels * 2), in_channels,kernel_size=kernel_size,padding=padding)

    def forward(self, reconstruction_features, segmentation_features):
        # Balance unit
        concat_features = torch.cat((reconstruction_features, segmentation_features), dim=1)
        balance_tensor = torch.sigmoid(self.balance_conv1(concat_features))
        balanced_output = self.balance_conv2(balance_tensor * reconstruction_features + (1 - balance_tensor) * segmentation_features)
        # Conv-deconvolution layers for spatial attention
        res_block = torch.sigmoid(self.residual_block(balanced_output))
        # Generate gated features
        gated_rec_features = (1 + res_block) * reconstruction_features
        gated_segmentation_features = (1 + res_block) * segmentation_features
        # Concatenate and apply convolutional layer
        concatenated_features = torch.cat((gated_rec_features, gated_segmentation_features), dim=1)
        output = self.fc(concatenated_features)
        return output
    

class ResidualBlock(nn.Module):
    def __init__(self, input_channels, output_channels, stride=1):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.stride = stride
        self.bn1 = nn.BatchNorm2d(input_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(input_channels, int(output_channels/4),  1,1, bias = False)
        self.bn2 = nn.BatchNorm2d(int(output_channels/4))
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(int(output_channels/4), int(output_channels/4), 3, stride, padding = 1, bias = False)
        self.bn3 = nn.BatchNorm2d(int(output_channels/4))
        self.relu = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(int(output_channels/4), output_channels, 1, 1, bias = False)
        self.conv4 = nn.Conv2d(input_channels, output_channels , 1, stride, bias = False)
        
    def forward(self, x):
        residual = x
        out = self.bn1(x)
        out1 = self.relu(out)
        out = self.conv1(out1)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)
        if (self.input_channels != self.output_channels) or (self.stride !=1 ):
            residual = self.conv4(out1)
        out += residual
        return out