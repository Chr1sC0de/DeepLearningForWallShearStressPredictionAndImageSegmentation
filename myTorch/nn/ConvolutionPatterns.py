try:
    from .. import nn as _nn
except ImportError:
    from myTorch import nn as _nn

import abc as _abc

operation_order = [
    'convolution_layer', 'normalization_layer', 'activation_layer']


class ConvPatternBase(_nn.Layer):

    reverse_order = False

    def __init__(self, *args, **kwargs):
        super(ConvPatternBase, self).__init__(*args, **kwargs)
        self.convolution_layer = self.convolution_constructor(
            *self.args, **self.kwargs)
        self.normalization_layer = self.normalization_constructor(
            self.args[self.in_norm_act]
        )
        self.activation_layer = self.activation_constructor(
            self.args[self.in_norm_act]
        )

        self.out_channels = self.convolution_layer.out_channels

    def main_forward(self, x):
        for method in self.iterable_ops:
            x = getattr(self, method)(x)
        return x

    @property
    def iterable_ops(self):
        if self.reverse_order:
            return reversed(operation_order)
        return operation_order

    @property
    def in_norm_act(self):
        if not self.reverse_order:
            return 1
        return 0

    @_abc.abstractmethod
    def convolution_constructor(self, *args, **kwargs):
        NotImplemented

    @_abc.abstractmethod
    def normalization_constructor(self, *args, **kwargs):
        NotImplemented

    @_abc.abstractmethod
    def activation_constructor(self, *args, **kwargs):
        NotImplemented


class _ForwardConvPattern(ConvPatternBase):
    normalization_constructor = _nn.InstanceNorm2d
    activation_constructor = _nn.ReLU


class ConvNormAct2d(_ForwardConvPattern):
    convolution_constructor = _nn.Conv2d

class ConvNormAct3d(ConvPatternBase):
    convolution_constructor = _nn.Conv3d
    normalization_constructor = _nn.InstanceNorm3d
    activation_constructor = _nn.ReLU

class DWConvNormAct2d(_ForwardConvPattern):
    convolution_constructor = _nn.DWConv2d


class SepConvNormAct2d(_ForwardConvPattern):
    convolution_constructor = _nn.SepConv2d


class _ReverseConvPattern(_ForwardConvPattern):
    reverse_order = True


class ActNormConv2d(_ReverseConvPattern):
    convolution_constructor = _nn.Conv2d


class ActNormDWConv2d(_ReverseConvPattern):
    convolution_constructor = _nn.DWConv2d


class ActNormSepConv2d(_ReverseConvPattern):
    convolution_constructor = _nn.SepConv2d

