import torch as _torch
try:
    from .. import nn as _nn
except ImportError:
    from myTorch import nn as _nn

import abc as _abc


class NormBase(_nn.Layer):

    def __init__(self, *args, **kwargs):
        super(NormBase, self).__init__(*args, **kwargs)
        self.layer = self.layer_constructor(
            *self.args, **self.kwargs
            )
        self.out_channels = self.args[0]

    def main_forward(self, x):
        return self.layer(x)

    @_abc.abstractmethod
    def layer_constructor(self, *args, **kwargs):
        NotImplementedError


class _BINormCommon(NormBase):
    default_kwargs = dict(
        track_running_stats=False,
        eps=0.001, momentum=0.99, affine=True)


class BatchNorm1d(_BINormCommon):
    layer_constructor = _nn.LayerPartial(
        _torch.nn.BatchNorm1d, **_BINormCommon.default_kwargs)


class BatchNorm2d(_BINormCommon):
    layer_constructor = _nn.LayerPartial(
        _torch.nn.BatchNorm2d, **_BINormCommon.default_kwargs)


class BatchNorm3d(_BINormCommon):
    layer_constructor = _nn.LayerPartial(
        _torch.nn.BatchNorm3d, **_BINormCommon.default_kwargs)


class InstanceNorm1d(_BINormCommon):
    layer_constructor = _nn.LayerPartial(
        _torch.nn.InstanceNorm1d, **_BINormCommon.default_kwargs)


class InstanceNorm2d(_BINormCommon):
    layer_constructor = _nn.LayerPartial(
        _torch.nn.InstanceNorm2d, **_BINormCommon.default_kwargs)


class InstanceNorm3d(_BINormCommon):
    layer_constructor = _nn.LayerPartial(
        _torch.nn.InstanceNorm3d, **_BINormCommon.default_kwargs)


class GroupNorm(NormBase):
    layer_constructor = _torch.nn.GroupNorm

    def parse_arguments(self, *args, **kwargs):
        super(NormBase, self).parse_arguments(*args, **kwargs)

        if len(self.args) == 1:
            self.args = list(self.args)
            self.args.append(self.args[0])