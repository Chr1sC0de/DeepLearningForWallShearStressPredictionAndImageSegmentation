try:
    from .. import nn as _nn
    from .. import Models as _Models
    from .. import Cat as _Cat
except ImportError:
    from myTorch import nn as _nn
    from myTorch import Models as _Models
    from myTorch import Cat as _Cat
import torch as _torch


class _NameSetter:

    def __init__(self, name):
        self.name = name

    def __get__(self, obj, obj_type):
        return obj._level_prefix % (obj.i, self.name)


class UnetBuilder(_nn.Layer):

    bottle_neck_connector = _nn.AvgPool2d
    decoder_constructor = _nn.VGGBlock
    upsample_constructor = _nn.UpsampleConv2d

    upsample_name = _NameSetter('upsample')
    concat_name = _NameSetter('concat')
    conv_name = _NameSetter('convolve')

    _level_prefix = 'level_%d_%s'

    output_activation = None

    output_conv_constructor = _torch.nn.Conv2d

    def __init__(self, encoder, output_classes, kernel_size, bottleneck=None):
        super(UnetBuilder, self).__init__()

        self.kernel_size = kernel_size

        self.encoder_callback = encoder
        self.encoder_callback.track_skip = True

        self.assign_bottleneck(bottleneck)

        up_channels = self.bottleneck.out_channels

        for self.i, concat_layer in self.iterable_layers():

            up_layer = self.upsample_constructor(
                up_channels, up_channels//2, kernel_size
            )
            setattr(self, self.upsample_name, up_layer)

            cat_layer = _Cat(up_layer, concat_layer)
            cat_channels = cat_layer.out_channels
            setattr(self, self.concat_name, cat_layer)

            decode_layer = self.decoder_constructor(
                cat_channels, cat_channels//2, kernel_size)
            setattr(self, self.conv_name, decode_layer)

            up_channels = decode_layer.out_channels

        self.pw_conv = self.output_conv_constructor(
            decode_layer.out_channels, output_classes, 1, bias=False)

    def decoder_forward(self, x):
        iterable_layers = enumerate(
            reversed(self.skip_layers)
        )
        for self.i, concat_layer in iterable_layers:
            x = getattr(self, self.upsample_name)(x)
            x = getattr(self, self.concat_name)(x, concat_layer)
            x = getattr(self, self.conv_name)(x)

        return x

    def main_forward(self, x):
        x = self.encoder_callback(x)
        x = self.bottleneck_connector(x)
        x = self.bottleneck(x)
        x = self.decoder_forward(x)
        x = self.pw_conv(x)
        if self.output_activation is not None:
            x = self.output_activation(x)
        return x

    def iterable_layers(self):
        """iterable_layers

        Given the current state of the model and encoder extract the
        concatenable layers.
        """
        return enumerate(reversed(self.encoder_callback.concatenable_layers))

    def assign_bottleneck(self, bottleneck):
        if hasattr(self.encoder_callback, 'connector_constructor'):
            self.bottleneck_connector = \
                self.encoder_callback.connector_constructor(
                    self.encoder_callback)
        else:
            self.bottleneck_connector = self.bottle_neck_connector(
                self.encoder_callback.out_channels, 2)
        if bottleneck is None:
            self.bottleneck = \
                self.encoder_callback.layer_constructor(
                    self.bottleneck_connector,
                    self.encoder_callback.out_channels*2,
                    self.kernel_size,
                    stride=1)
        else:
            self.bottleneck = bottleneck(
                    self.bottleneck_connector,
                    self.encoder_callback.out_channels*2,
                    self.kernel_size,
                    stride=1)

    @property
    def skip_layers(self):
        return self.encoder_callback.skip_layers


class Unet(UnetBuilder):
    output_activation = None

    def __init__(
            self, in_channels, num_classes, kernel_size, n_pool=4,
            base_channels=64, layer_repeats=1, output_activation=None):
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
        layers = [layer_repeats]*n_pool
        encoder = _Models.VGGNet(
            in_channels, base_channels, kernel_size,
            layers=layers)
        super(Unet, self).__init__(encoder, num_classes, kernel_size)
        if output_activation is not None:
            self.output_activation = output_activation

class Unet3d(UnetBuilder):
    output_activation = None
    bottle_neck_connector = _nn.AvgPool3d
    decoder_constructor = _nn.VGGBlock3d
    upsample_constructor = _nn.UpsampleConv3d
    output_conv_constructor = _torch.nn.Conv3d
    def __init__(
            self, in_channels, num_classes, kernel_size, n_pool=4,
            base_channels=64, layer_repeats=1, output_activation=None):
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
        layers = [layer_repeats]*n_pool
        encoder = _Models.VGGNet3D(
            in_channels, base_channels, kernel_size,
            layers=layers)
        super(Unet3d, self).__init__(encoder, num_classes, kernel_size)
        if output_activation is not None:
            self.output_activation = output_activation


class ResnetUnet(UnetBuilder):
    output_activation = None

    def __init__(
        self, in_channels, num_classes, kernel_size=3,
            layers=[3, 3, 5, 2],
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
        encoder = _Models.ResNet(
            in_channels, base_channels, kernel_size,
            layers=encoder_layers)
        n_bottleneck = layers[-1]

        class ResBottle(_nn.RepeatedLayers):
            layer_constructor = encoder.layer_constructor
            _layer_dict = dict(
                stride=[[1, 1]] * n_bottleneck,
            )

        super(ResnetUnet, self).__init__(
            encoder, num_classes, kernel_size, bottleneck=ResBottle)
        if output_activation is not None:
            self.output_activation = output_activation


class ResnetBottleneckUnet(UnetBuilder):
    output_activation = None

    def __init__(
        self, in_channels, num_classes, kernel_size=3,
            layers=[3, 3, 5, 2],
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
        encoder = _Models.ResNetBottleNeck(
            in_channels, base_channels, kernel_size,
            layers=encoder_layers)
        n_bottleneck = layers[-1]

        class ResBottle(_nn.RepeatedLayers):
            layer_constructor = encoder.layer_constructor
            _layer_dict = dict(
                stride=[[1, 1]] * n_bottleneck,
            )

        super(ResnetBottleneckUnet, self).__init__(
            encoder, num_classes, kernel_size, bottleneck=ResBottle)
        if output_activation is not None:
            self.output_activation = output_activation


if __name__ == '__main__':
    import torch as _torch
    from torch.utils.tensorboard import SummaryWriter
    filterSize = 3
    strides = 1
    dilationRate = 1

    writer = SummaryWriter()

    kwDict = dict(
        stride=strides,
        dilation=dilationRate,
        padding='same'
    )

    argsTuple = (filterSize, strides, dilationRate)

    XYZ = _torch.meshgrid(*[_torch.arange(0, 16)]*2)
    testObj = _torch.stack(XYZ, dim=0)
    testObj = testObj.unsqueeze(0).float()

    print(testObj.shape)

    encoder = Unet(2, 1, 3)

    writer.add_graph(encoder, testObj)
    writer.flush()
    writer.close()
    print('done')