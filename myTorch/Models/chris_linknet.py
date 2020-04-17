from .EncoderBottleNeckHead import EncodeDecodeBase
try:
    from myTorch import nn, Models
except ImportError:
    from .. import nn, Models
import torch


class LinkNetBuilder(EncodeDecodeBase):
    decoder_constructor = nn.LinkNetDecoder
    upsample_constructor = torch.nn.UpsamplingBilinear2d

    def decoder_forward(self, x):
        iterable_layers = enumerate(
            reversed(self.skip_layers)
        )
        for self.i, add_layer in iterable_layers:
            x = getattr(self, self.upsample_name)(x)
            x = getattr(self, self.conv_name)(x)
            x = x + add_layer

        return x

    def construct_top_down_path(self, in_channels):

        iterable_layers = list(self.iterable_layers())

        self.number_of_scales = len(iterable_layers)

        for self.i, addable_layer in iterable_layers:
            up_layer = self.upsample_constructor(scale_factor=2)
            setattr(self, self.upsample_name, up_layer)
            conv_layer = self.decoder_constructor(in_channels, in_channels//2, 3)
            setattr(self, self.conv_name, conv_layer)
            in_channels = in_channels//2

        return conv_layer


class ResNetLinkNet(LinkNetBuilder):
    output_activation = None

    def __init__(
        self, in_channels, num_classes, kernel_size=3,
            layers=[2, 1, 1, 1],
            base_channels=64,
            output_activation=None):
        """ Build a Resnet Unet model

        https://towardsdatascience.com/u-nets-with-resnet-encoders-and-cross-connections-d8ba94125a2c
        The model however uses instance normalization before each activation
        and uses Average pooling over max pooling.

        Args:
            in_channels ([int])
            num_classes ([int]): the number of output channels
            kernel_size ([int])]
        Kwargs:
            base_channels ([int]): The total number
                of channels the networks is initialized with
            output_activation([_nn.Layer,_torch.nn.functional]):
                activation function of form f(x)
        """
        encoder_layers = layers[:-1]
        encoder = Models.ResNet(
            in_channels, base_channels, kernel_size,
            layers=encoder_layers)
        n_bottleneck = layers[-1]

        class ResBottle(nn.RepeatedLayers):
            layer_constructor = encoder.layer_constructor
            _layer_dict = dict(
                stride=[[1, 1]] * n_bottleneck,
            )

        super(ResNetLinkNet, self).__init__(
            encoder, num_classes, kernel_size, bottleneck=ResBottle)

        self.output_activation = output_activation


def rn34LinkNet(in_channels, num_classes, kernel_size, **kwargs):
    return ResNetLinkNet(
        in_channels, num_classes, kernel_size=kernel_size, 
        layers=[3, 3, 5, 2], **kwargs
        )


class VGGLinkNet(LinkNetBuilder):
    output_activation = None

    def __init__(
            self, in_channels, num_classes, kernel_size, n_pool=4,
            base_channels=64, output_activation=None):
        """__init__ Build a vanilla Unet model

        https://arxiv.org/abs/1505.04597
        The model however uses instance normalization before each activation
        and uses Average pooling over max pooling.

        Args:
            in_channels ([type]): [description]
            num_classes ([type]): [description]
            kernel_size ([type]): [description]
            n_pool (int, optional): [description]. Defaults to 4.
            base_channels (int, optional): [description]. Defaults to 64.
            output_activation ([type], optional): [description]. Defaults to None.
        """
        layers = [1]*n_pool
        encoder = Models.VGGNet(
            in_channels, base_channels, kernel_size,
            layers=layers)
        super(VGGLinkNet, self).__init__(encoder, num_classes, kernel_size)
        if output_activation is not None:
            self.output_activation = output_activation
