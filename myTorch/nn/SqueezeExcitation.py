try:
    from .. import nn as _nn
except ImportError:
    from myTorch import nn as _nn

import torch as _torch


class SEBlock(_nn.Layer):
    """SEBlock

    Squeeze and excitation blocks

    https://arxiv.org/pdf/1709.01507.pdf
    """

    def __init__(self, *args, scale_factor=16, **kwargs):
        super(SEBlock, self).__init__(
            *args, scale_factor=scale_factor, **kwargs)


        self.in_channels = args[0] if isinstance(args[0], int) else args[0].out_channels

        self.out_channels = self.in_channels

        self.scaled_channels = max(self.in_channels//scale_factor, 1)

        self.global_pooling = _torch.nn.functional.avg_pool2d

        self.fc_1 = _torch.nn.Linear(
            self.in_channels, self.scaled_channels, bias=False)

        self.activation_layer = _torch.nn.ReLU()

        self.fc_2 = _torch.nn.Linear(
            self.scaled_channels, self.out_channels, bias=False)

        self.sigmoid_layer = _torch.nn.Sigmoid()

    def main_forward(self, x):
        _, _, *hw = x.shape
        x = self.global_pooling(x, hw)
        x = x.permute(0, 2, 3, 1)

        x = self.fc_1(x)
        x = self.activation_layer(x)
        x = self.fc_2(x)
        x = self.sigmoid_layer(x)
        x = x.permute(0, 3, 1, 2)

        return x

class CSEBlock(_torch.nn.Module):
    """CSEBlock

    Channel Squeeze and spatial excitation

    https://arxiv.org/pdf/1803.02579.pdf
    """
    def __init__(self, in_channels):
        super(CSEBlock, self).__init__()
        self.conv = _torch.nn.Conv2d(in_channels, 1, 1, bias=False)
        self.activation = _torch.nn.Sigmoid()
        self.out_channels = 1

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x

#spatial squeeze and channel excite

class SEResBlock(_nn.ResidualBlock):

    def __init__(self, *args, scale_factor=16, **kwargs):
        super(SEResBlock, self).__init__(*args, **kwargs)
        self.se_block = SEBlock(self.out_channels, scale_factor=scale_factor)

    def post_forward(self, x):
        x_scale = self.se_block(x)
        x = x * x_scale
        return super(SEResBlock, self).post_forward(x)


class SEVGGBlock(_nn.VGGBlock):
    def __init__(self, *args, scale_factor=16, **kwargs):
        super(SEVGGBlock, self).__init__(*args, **kwargs)
        self.se_block = SEBlock(self.out_channels, scale_factor=scale_factor)

    def post_forward(self, x):
        x_scale = self.se_block(x)
        x = x * x_scale
        return super(SEVGGBlock, self).post_forward(x)

# channel squeeze spatial excite

class CSEVGGBlock(_nn.VGGBlock):
    def __init__(self, *args, **kwargs):
        super(CSEVGGBlock, self).__init__(*args, **kwargs)
        self.cse_block = CSEBlock(self.out_channels)

    def post_forward(self, x):
        x_scale = self.cse_block(x)
        x = x * x_scale
        return super(CSEVGGBlock, self).post_forward(x)


# concurrent

class CSESEVGGBlock(_nn.VGGBlock):
    def __init__(self, *args, scale_factor=16, **kwargs):
        super(CSESEVGGBlock, self).__init__(*args, **kwargs)
        self.cse_block = CSEBlock(self.out_channels)
        self.se_block = SEBlock(self.out_channels, scale_factor=scale_factor)

    def post_forward(self, x):
        x_excite_spatial = self.cse_block(x)
        x_excite_channel = self.se_block(x)
        x = x * x_excite_spatial + x * x_excite_channel
        return super(CSESEVGGBlock, self).post_forward(x)



