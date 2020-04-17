try:
    from .. import nn as _nn
except ImportError:
    from myTorch import nn as _nn
import torch as _torch


class _UpsampleConv2dBase(_nn.ConvNormAct2d):
    upsample_constructor = _nn.LayerPartial(
        _torch.nn.Upsample, scale_factor=2, align_corners=True,
        mode='bilinear')

    def __init__(self, *args, scale_factor=2, **kwargs):
        super(_UpsampleConv2dBase, self).__init__(*args, **kwargs)
        self.upsampling_layer = self.upsample_constructor(
            scale_factor=scale_factor
        )
        self.out_channels = self.convolution_layer.out_channels

    def main_forward(self, x):
        x = self.upsampling_layer(x)
        x = super(_UpsampleConv2dBase, self).main_forward(x)
        return x

class _UpsampleConv3dBase(_nn.ConvNormAct3d):
    upsample_constructor = _nn.LayerPartial(
        _torch.nn.Upsample, scale_factor=2, align_corners=True,
        mode='trilinear')

    def __init__(self, *args, scale_factor=2, **kwargs):
        super(_UpsampleConv3dBase, self).__init__(*args, **kwargs)
        self.upsampling_layer = self.upsample_constructor(
            scale_factor=scale_factor
        )
        self.out_channels = self.convolution_layer.out_channels

    def main_forward(self, x):
        x = self.upsampling_layer(x)
        x = super(_UpsampleConv3dBase, self).main_forward(x)
        return x


class UpsampleConv2d(_UpsampleConv2dBase):
    convolution_constructor = _nn.Conv2d

class UpsampleConv3d(_UpsampleConv3dBase):
    convolution_constructor = _nn.Conv3d


class DWUpsampleConv2d(_UpsampleConv2dBase):
    convolution_constructor = _nn.DWConv2d


class SepUpsampleConv2d(_UpsampleConv2dBase):
    convolution_constructor = _nn.SepConv2d


if __name__ == "__main__":
    import torch
    image = torch.zeros([1, 1, 25, 25])
    upsampled = UpsampleConv2d(image, 10, 3, scale_factor=4)(image)
    print(image.shape)
    print(upsampled.shape)
