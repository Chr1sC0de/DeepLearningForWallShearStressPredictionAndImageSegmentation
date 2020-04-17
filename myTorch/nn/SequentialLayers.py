try:
    from .. import nn as _nn
except ImportError:
    from myTorch import nn as _nn
import abc as _abc
import itertools as _itertools


class Sequential(_nn.Layer):

    def __init__(self, *layers):
        super(Sequential, self).__init__()
        self.n_layers = len(layers)
        self.layer_registry = layers
        self.layer_names = []
        for i, layer in enumerate(self.layer_registry):
            name = f'layer_{i}'
            self.layer_names.append(name)
            setattr(
                self, name, layer
            )
        self.out_channels = layers[-1].out_channels

    def main_forward(self, x, **kwargs):
        for layer in self.layer_registry:
            x = layer(x)
        return x


class RepeatedLayers(Sequential):
    _layer_dict = dict()

    def __init__(self, *args, layer_dict={}, **kwargs):
        self.parse_arguments(*args, **kwargs)
        self.layer_registry = []
        self.layer_dict = self._layer_dict.copy()
        self.layer_dict.update(layer_dict)
        self.in_channels = self.args[0]
        self.check_kwarg_dict_same_length()
        self.make_repeated_keys_n_values()
        self.construct_layer_registry()
        super(RepeatedLayers, self).__init__(*self.layer_registry)

    def construct_layer_registry(self):
        for i, (keys, zipped) in enumerate(
                zip(self.repeated_keys, self.zipped_values)):
            self.kwargs.update(dict(zip(keys, zipped)))
            if i == 1:
                self.args = list(self.args)
                self.args[0] = self.args[1]

            self.layer_registry.append(
                self.layer_constructor(
                    *self.args, **self.kwargs
                )
            )

    def make_repeated_keys_n_values(self):
        self.repeated_keys = _itertools.repeat(
            self.layer_dict.keys(), self.n_repeat)
        self.zipped_values = zip(*self.layer_dict.values())

    def check_kwarg_dict_same_length(self):
        assert len(self.layer_dict)

        key_0 = list(self.layer_dict.keys())[0]
        self.n_repeat = len(self.layer_dict[key_0])
        for key, item in self.layer_dict.items():
            assert len(item) == self.n_repeat

    @_abc.abstractmethod
    def layer_constructor(self, *args, **kwargs):
        NotImplemented


class VGGBlock(RepeatedLayers):
    '''
    implementation of the basic VGG block
    https://arxiv.org/pdf/1409.1556.pdf
    '''
    layer_constructor = _nn.ConvNormAct2d
    _layer_dict = dict(
        stride=[1, 1],
        dilation=[1, 1]
    )


class VGGBlock3d(RepeatedLayers):
    '''
    implementation of the basic VGG block
    https://arxiv.org/pdf/1409.1556.pdf
    '''
    layer_constructor = _nn.ConvNormAct3d
    _layer_dict = dict(
        stride=[1, 1],
        dilation=[1, 1]
    )


class DWVGGBlock(VGGBlock):
    '''
    implementation of the depthwise VGG block
    https://arxiv.org/pdf/1409.1556.pdf
    '''
    layer_constructor = _nn.DWConvNormAct2d


class SepVGGBlock(VGGBlock):
    '''
    implementation of the seperable VGG block
    https://arxiv.org/pdf/1409.1556.pdf
    '''
    layer_constructor = _nn.SepConvNormAct2d


class LinkNetDecoder(_nn.Layer):
    def __init__(self, in_channels, out_channels, kernel_size, rescale_factor=4, **kwargs):
        super(LinkNetDecoder, self).__init__(
            in_channels, out_channels, kernel_size, **kwargs)
        scaled_in = in_channels//4
        self.conv1 = _nn.ConvNormAct2d(in_channels, scaled_in, 1, **kwargs)
        self.conv2 = _nn.ConvNormAct2d(self.conv1, scaled_in, kernel_size, **kwargs)
        self.conv3 = _nn.ConvNormAct2d(self.conv2, out_channels, 1, **kwargs)
        self.out_channels = out_channels

    def main_forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x