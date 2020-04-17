import torch as _torch

try:
    from .. import nn as _nn
except ImportError:
    from myTorch import nn as _nn


class PyramidPool(_nn.Layer):
    conv_constructor = _nn.ConvNormAct2d

    def __init__(self, in_channels, out_channels, kernel_sizes=[1, 3, 5, 7], **kwargs):
        super(PyramidPool, self).__init__(in_channels, out_channels, **kwargs)
        assert out_channels % len(kernel_sizes) == 0, 'out_channels mut be divisible by number of kernels'
        self.pyramid_layers = []
        n_kernels = len(kernel_sizes)

        for i, kernel in enumerate(kernel_sizes):
            self.pyramid_layers.append(
                self.conv_constructor(in_channels, out_channels//n_kernels, kernel, **kwargs)
            )
            setattr(
                self, 'layer_%d' % i, self.pyramid_layers[-1]
            )
        self.out_channels = out_channels

    def main_forward(self, x):
        to_concat = []
        for layer in self.pyramid_layers:
            to_concat.append(
                layer(x)
            )
        return _torch.cat(to_concat, dim=1)


class AtrousPyramidPoolingBase(_nn.RepeatedLayers):
    layer_dict = dict(
        dilation=[1, 6, 12, 18],
        stride=[1, 1, 1, 1]
    )

    interpolation_mode = 'bilinear'
    billinear_layer = None

    def __init__(self, *args, **kwargs):
        super(AtrousPyramidPoolingBase, self).__init__(*args, **kwargs)

        n_layers = len(next(iter(self.layer_dict.values())))

        if self.billinear_layer is not None:
            billinear_channels = self.billinear_layer.shape[1]
        else:
            billinear_channels = 0

        self.pw_conv = _torch.nn.Conv2d(
            self.args[0]*n_layers + billinear_channels, args[1], 1, bias=False
        )

    def main_forward(self, x):

        if self.billinear_layer is not None:
            _, _, h, w = x.shape
            concat_layers = [
                _torch.nn.functional.interpolate(
                    self.billinear_layer,
                    size=(h, w),
                    mode=self.interpolation_mode
                )
            ]
        else:
            concat_layers = []

        for name in self.layer_names:
            concat_layers.append(
                getattr(self, name)(x))

        x = _torch.cat(concat_layers, dim=1)
        return self.pw_conv(x)


class ASPP(AtrousPyramidPoolingBase):
    layer_constructor = _nn.ConvNormAct2d


class DWASPP(AtrousPyramidPoolingBase):
    layer_constructor = _nn.DWConvNormAct2d


class SepASPP(AtrousPyramidPoolingBase):
    layer_constructor = _nn.SepConvNormAct2d
