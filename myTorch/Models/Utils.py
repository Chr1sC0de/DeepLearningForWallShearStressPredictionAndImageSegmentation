"""
Utility functions for models

"""

import torch
import torch.nn as _nn
import numpy as numpy
import torch.nn.functional as F

class ResNetBlock(_nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, groups=1, bias=False):
        super(ResNetBlock, self).__init__()
        self.conv1 = _nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=bias)
        self.bn1 = _nn.BatchNorm2d(out_planes)
        self.relu = _nn.ReLU(inplace=True)
        self.conv2 = _nn.Conv2d(out_planes, out_planes, kernel_size, 1, padding, groups=groups, bias=bias)
        self.bn2 = _nn.BatchNorm2d(out_planes)
        self.downsample = None
        if stride > 1:
            self.downsample = _nn.Sequential(_nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False),
                            _nn.BatchNorm2d(out_planes),)

    def forward(self, x):       

        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample:
            residual = self.downsample(x)
        
        out += residual
        out = self.relu(out)

        return out
class Encoder(_nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, groups=1, bias=False):
        super(Encoder, self).__init__()
        self.block1 = ResNetBlock(in_planes, out_planes, kernel_size, stride, padding, groups, bias)
        self.block2 = ResNetBlock(out_planes, out_planes, kernel_size, 1, padding, groups, bias)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)

        return x

class Decoder(_nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=False):
        super(Decoder, self).__init__()
        self.conv1 = _nn.Sequential(_nn.Conv2d(in_planes, in_planes//4, 1, 1, 0, bias=bias),
                                _nn.BatchNorm2d(in_planes//4),
                                _nn.ReLU(inplace=True),)
        self.tp_conv = _nn.Sequential(_nn.ConvTranspose2d(in_planes//4, in_planes//4, kernel_size, stride, padding, output_padding, bias=bias),
                                _nn.BatchNorm2d(in_planes//4),
                                _nn.ReLU(inplace=True),)
        self.conv2 = _nn.Sequential(_nn.Conv2d(in_planes//4, out_planes, 1, 1, 0, bias=bias),
                                _nn.BatchNorm2d(out_planes),
                                _nn.ReLU(inplace=True),)

    def forward(self, x):
        x = self.conv1(x)
        x = self.tp_conv(x)
        x = self.conv2(x)

        return x

