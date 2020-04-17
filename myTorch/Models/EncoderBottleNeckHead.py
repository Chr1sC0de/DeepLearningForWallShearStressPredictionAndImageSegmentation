try:
    from myTorch import nn as _nn
except ImportError:
    from .. import nn as _nn
import torch as _torch


class _NameSetter:

    def __init__(self, name):
        self.name = name

    def __get__(self, obj, obj_type):
        return obj._level_prefix % (obj.i, self.name)


class EncodeDecodeBase(_nn.Layer):

    decoder_constructor = _nn.VGGBlock
    upsample_constructor = _nn.UpsampleConv2d

    upsample_name = _NameSetter('upsample')
    concat_name = _NameSetter('concat')
    conv_name = _NameSetter('convolve')

    _level_prefix = 'level_%d_%s'

    output_activation = None

    def __init__(self, encoder, output_classes, kernel_size, bottleneck=None):
        super(EncodeDecodeBase, self).__init__()

        self.kernel_size = kernel_size

        self.encoder_callback = encoder
        self.encoder_callback.track_skip = True

        self.assign_bottleneck(bottleneck)

        up_channels = self.bottleneck.out_channels

        final_layer = self.construct_top_down_path(up_channels)

        self.pw_conv = _torch.nn.Conv2d(
            final_layer.out_channels, output_classes, 1, bias=False)

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
