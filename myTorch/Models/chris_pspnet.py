from .EncoderBottleNeckHead import EncodeDecodeBase
try:
    from myTorch import nn, Models
except ImportError:
    from .. import nn, Models
import torch


class PSPNetBuilder(nn.Layer):

    def __init__(self, in_channels, n_classes, kernel_size, layers=[3, 3, 5, 2],
            base_channels=64, output_activation=None,**kwargs):
        super(PSPNetBuilder, self).__init__()
        self.encoder = Models.ResNet(
            in_channels, base_channels, kernel_size, layers=layers,
            **kwargs)

        encoder_channels = self.encoder.out_channels

        self.connector = self.encoder.connector_constructor(self.encoder)
            
        self.bottleneck = nn.PyramidPool(
            encoder_channels, encoder_channels*2
        )

        self.upsample = torch.nn.UpsamplingBilinear2d(scale_factor=2)

        self.upsample_layers = []

        n_upsamples = len(layers) - 1

        in_channels = self.bottleneck.out_channels + self.encoder.out_channels

        for i in range(n_upsamples):
            self.upsample_layers.append(
                nn.UpsampleConv2d(in_channels, in_channels//2, kernel_size)
            )
            setattr(self, 'upsample_%d' % i, self.upsample_layers[-1])

            in_channels = in_channels//2

        self.pw_conv = torch.nn.Conv2d(
            self.upsample_layers[-1].out_channels, n_classes, 1, bias=False
        )

        self.output_activation = output_activation

        self.out_channels = n_classes

    def main_forward(self, x):
        x_concat = self.encoder(x)
        x = self.connector(x_concat)
        x = self.bottleneck(x)
        x = self.upsample(x)
        x = torch.cat(
            [x, x_concat], dim=1
        )
        for layer in self.upsample_layers:
            x = layer(x)
        
        x = self.pw_conv(x)
        if self.output_activation is not None:
            x = self.output_activation(x)
        return x

class UPSPNet(PSPNetBuilder):
    feature_constructor = Models.VGGNet
    def __init__(self, *args, layers=[1, 1, 1, 1], **kwargs):
        super(UPSPNet, self).__init__(
            *args, layers=layers, **kwargs
        )

class ResNetPSPNet(PSPNetBuilder):
    feature_constructor = Models.ResNet

class ResNetBottleNeckPSPNet(PSPNetBuilder):
    feature_constructor = Models.ResNetBottleNeck




