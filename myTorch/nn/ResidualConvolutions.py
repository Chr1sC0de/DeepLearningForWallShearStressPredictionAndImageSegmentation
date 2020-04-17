try:
    from .. import nn as _nn
except ImportError:
    from myTorch import nn as _nn

import torch as _torch


class ResidualBase(_nn.RepeatedLayers):
    pool_constructor = _torch.nn.AvgPool2d

    def pre_forward(self, x, **kwargs):
        self.x_residual = x
        return x

    def post_forward(self, x):
        self.x_residual = self.spatial_controller(x, self.x_residual)
        self.x_residual = self.channel_controller(x, self.x_residual)
        return x + self.x_residual

    def spatial_controller(self, x, x_residual):
        _, _, x_h, _ = x.shape
        _, _, x_residual_h, _ = x_residual.shape

        if not hasattr(self, 'scaling_factor'):

            if x_h == x_residual_h:
                return x_residual

            if x_residual_h > x_h:
                x_exponent = _nn._findExponent(x_h)
                x_in_exponent = _nn._findExponent(x_residual_h)

            self.scaling_factor = x_in_exponent - x_exponent

        if self.scaling_factor != 0:
            if not hasattr(self, '_poolConcat'):
                self._poolConcat = self.pool_constructor(2)

            for _ in range(self.scaling_factor):
                x_residual = self._poolConcat(x_residual)

        return x_residual

    def channel_controller(self, x, x_residual):

        _, x_c, _, _ = x.shape

        _, x_residual_c, _, _ = x_residual.shape

        if x_residual_c == x_c:
            return x_residual

        if x_residual_c < x_c:
            diff = x_c - x_residual_c
            new_x = _torch.zeros_like(x[:, 0: diff, :, :])
            new_x = _torch.cat([x_residual, new_x], dim=1)
            return new_x

        if x_residual_c > x_c:
            return x_residual[:, 0:x_c, :, :]


class ResidualBlock(ResidualBase):
    layer_constructor = _nn.ActNormConv2d
    _layer_dict = dict(
        stride=[1, 1],
        dilation=[1, 1]
    )

    '''
    implementation of the following paper
    https://arxiv.org/pdf/1512.03385.pdf
    '''


class BottleNeckResidualBlock(ResidualBase):
    layer_constructor = _nn.ActNormConv2d
    _layer_dict = dict(
        stride=[1, ],
        dilation=[1, ]
    )

    def __init__(self, *args, scale=4, stride=1, **kwargs):
        args = list(args)

        original_in_channels = args[0]
        original_out_channels = args[1]

        args[0] = max(original_out_channels//scale, 1)
        args[1] = args[0]
        super(BottleNeckResidualBlock, self).__init__(*args, **kwargs)

        self.pw_conv_A = self.layer_constructor(
            original_in_channels, args[0], 1,
            padding=0, bias=False, stride=stride)

        self.pw_conv_B = self.layer_constructor(
            args[0], original_out_channels,
            1, padding=0, bias=False)

        self.out_channels = original_out_channels
        self.in_channels = original_in_channels

    def main_forward(self, x):
        x = self.pw_conv_A(x)
        x = super(BottleNeckResidualBlock, self).main_forward(x)
        x = self.pw_conv_B(x)
        return x


class DWResidualBlock(ResidualBlock):
    layer_constructor = _nn.ActNormDWConv2d

    '''
    depthwise implementation of the following paper
    https://arxiv.org/pdf/1512.03385.pdf
    '''


class SepResidualBlock(ResidualBlock):
    layer_constructor = _nn.ActNormSepConv2d

    '''
    seperable implementation of the following paper
    https://arxiv.org/pdf/1512.03385.pdf
    '''
