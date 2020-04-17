import torch as _torch
try:
    from . import Layer as _Layer
except ImportError:
    from myTorch.nn import Layer as _Layer
import abc as _abc


class _Common(_Layer):

    def __init__(self, in_channels, *args, **kwargs):
        super(_Common, self).__init__(*args, **kwargs)
        if isinstance(in_channels, int):
            self.out_channels = in_channels
        else:
            self.out_channels = in_channels.out_channels
        self.layer = self.layer_constructor(*self.args, **self.kwargs)

    def main_forward(self, x):
        return self.layer(x)

    @_abc.abstractmethod
    def layer_constructor(self, *args, **kwargs):
        NotImplementedError


class ELU(_Common):
    layer_constructor = _torch.nn.ELU


class Hardshrink(_Common):
    layer_constructor = _torch.nn.Hardshrink


class Hardtanh(_Common):
    layer_constructor = _torch.nn.Hardtanh


class LeakyReLU(_Common):
    layer_constructor = _torch.nn.LeakyReLU


class LogSigmoid(_Common):
    layer_constructor = _torch.nn.LogSigmoid


class PReLU(_Common):
    layer_constructor = _torch.nn.PReLU


class ReLU(_Common):
    layer_constructor = _torch.nn.ReLU


class ReLU6(_Common):
    layer_constructor = _torch.nn.ReLU6


class RReLU(_Common):
    layer_constructor = _torch.nn.RReLU


class SELU(_Common):
    layer_constructor = _torch.nn.SELU


class CELU(_Common):
    layer_constructor = _torch.nn.CELU


class Sigmoid(_Common):
    layer_constructor = _torch.nn.Sigmoid


class Softplus(_Common):
    layer_constructor = _torch.nn.Softplus


class Softshrink(_Common):
    layer_constructor = _torch.nn.Softshrink


class Softsign(_Common):
    layer_constructor = _torch.nn.Softsign


class Tanh(_Common):
    layer_constructor = _torch.nn.Tanh


class Tanhshrink(_Common):
    layer_constructor = _torch.nn.Tanhshrink


class Threshold(_Common):
    layer_constructor = _torch.nn.Threshold


class Softmin(_Common):
    layer_constructor = _torch.nn.Softmin


class Softmax(_Common):
    layer_constructor = _torch.nn.Softmax


class Softmax2d(_Common):
    layer_constructor = _torch.nn.Softmax2d


class LogSoftmax(_Common):
    layer_constructor = _torch.nn.LogSoftmax


class MaxPool2d(_Common):
    layer_constructor = _torch.nn.MaxPool2d


class AvgPool2d(_Common):
    layer_constructor = _torch.nn.AvgPool2d

class AvgPool3d(_Common):
    layer_constructor = _torch.nn.AvgPool3d
