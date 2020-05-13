import abc as _abc

import torch as _torch

try:
    from . import Layer as _Layer
    from . import Padding as _padding_mod
    from . import LayerPartial as _LayerConstructor
except ImportError:
    from myTorch.nn import Layer as _Layer
    from myTorch.nn import LayerPartial as _LayerConstructor
    import Padding as _padding_mod

_method_hash = dict(
    same='ZeroPad2d',
    constant='ReplicationPad2d',
    periodic_replication='PeriodicReplication2d'
)


def _findExponent(in_val, start=2):
    i = 0
    val = 2**i
    while val < in_val:
        i += 1
        val = start**i
    return i


class AssignFromLayer:
    def __init__(self, Layer):
        self.layer = Layer

    def __call__(self, obj, names):
        for name in names:
            setattr(obj, name, getattr(self.layer, name))


class ConvBase(_Layer):
    use_special_padding = False
    __slots__ = ['kernel_size', 'stride', 'dilation']
    default_padding_string = 'same'

    def __init__(self, *args, **kwargs):
        super(ConvBase, self).__init__(*args, **kwargs)
        self.layer = self.layer_constructor(*self.args, **self.kwargs)
        AssignFromLayer(self.layer)(self, self.__slots__)
        self.build_padder()

    def pre_forward(self, x, **kwargs):
        return self.pad_x(x, **kwargs)

    def main_forward(self, x, **kwargs):
        x = self.layer(x, **kwargs)
        return x

    def build_padder(self):
        if self.use_special_padding:
            self.padder = self.special_padding(
                self.kernel_size[0], self.stride[0], self.dilation[0])

    def pad_x(self, x, **kwargs):
        if self.use_special_padding:
            x = self.padder(x, **kwargs)
        return x

    def parse_arguments(self, *args, **kwargs):
        self.out_channels = args[1]
        padding = kwargs.get('padding', self.default_padding_string)
        if isinstance(padding, str):
            if padding in _method_hash.keys():
                self.use_special_padding = True
                self.special_padding = getattr(
                    _padding_mod, _method_hash[padding]
                )
                kwargs['padding'] = 0
        else:
            assert isinstance(padding, int)
        super(ConvBase, self).parse_arguments(*args, **kwargs)

    @_abc.abstractmethod
    def layer_constructor(self, x):
        NotImplemented


class Conv3dBase(_Layer):
    use_special_padding = False
    __slots__ = ['kernel_size', 'stride', 'dilation']
    padding_mode = 'replicate'

    def __init__(self, *args, **kwargs):
        super(Conv3dBase, self).__init__(*args, **kwargs)
        self.layer = self.layer_constructor(*self.args, **self.kwargs)
        AssignFromLayer(self.layer)(self, self.__slots__)

    def main_forward(self, x, **kwargs):
        x = self.layer(x, **kwargs)
        return x

    def parse_arguments(self, *args, **kwargs):
        self.out_channels = args[1]
        padding = kwargs.get('padding', self.padding_mode)
        if isinstance(padding, str):
            kwargs['padding'] = args[2]//2
            kwargs['padding_mode'] = padding
        else:
            assert isinstance(padding, int)
        super(Conv3dBase, self).parse_arguments(*args, **kwargs)

    @_abc.abstractmethod
    def layer_constructor(self, x):
        NotImplemented


class Conv2d(ConvBase):
    layer_constructor = _LayerConstructor(
        _torch.nn.Conv2d, bias=False
    )


class Conv3d(Conv3dBase):
    layer_constructor = _LayerConstructor(
        _torch.nn.Conv3d, bias=False
    )

class ConvTranspose2d(ConvBase):
    layer_constructor = _LayerConstructor(
        _torch.nn.ConvTranspose2d, bias=False, stride=2
    )


class ConcatConv(Conv2d):
    concat_layer = None
    _concat_downsampler = _torch.nn.AvgPool2d
    '''
    abstracted layer form of the CoordConv
    https://arxiv.org/pdf/1807.03247.pdf
    '''

    def __init__(self, in_channels, *args, **kwargs):
        assert self.concat_layer is not None, "run, set_concat"
        in_channels += self.concat_layer.shape[1]
        super(ConcatConv, self).__init__(
                in_channels, *args, **kwargs
        )
        self.size_checked = False

    @classmethod
    def set_concat(cls, toConcat):
        cls.concat_layer = toConcat

    def pre_forward(self, x):

        # if not hasattr(self, 'concat_layer'):
        if not self.size_checked:
            self.size_checked = True
            _, _, h_concat, _ = self.concat_layer.shape
            _, _, h_x, _ = x.shape
            toConcatExponent = _findExponent(h_concat)
            xExponent = _findExponent(h_x)
            scalingFactor = toConcatExponent - xExponent
            if scalingFactor != 0:
                if not hasattr(self, '_pool_concat'):
                    self._pool_concat = self._concat_downsampler(2)
                for _ in range(scalingFactor):
                    self.concat_layer = \
                        self._pool_concat(self.concat_layer)

        x = _torch.cat([x, self.concat_layer], dim=1)
        x = super().pre_forward(x)
        return x


class DWConv2d(Conv2d):
    def __init__(self, *args, **kwargs):
        super(DWConv2d, self).__init__(*args, groups=args[0], **kwargs)


class _layer_wrapper:
    def __init__(self, dw_conv, pw_conv):
        self.dw_conv = dw_conv
        self.pw_conv = pw_conv
        self.kernel_size = dw_conv.kernel_size
        self.stride = dw_conv.stride
        self.dilation = dw_conv.dilation

    def __call__(self, x):
        x = self.dw_conv(x)
        x = self.pw_conv(x)
        return x


class SepConv2d(ConvBase):
    '''
    implementation of depthwise seperable convolutions
    http://zpascal.net/cvpr2017/Chollet_Xception_Deep_Learning_CVPR_2017_paper.pdf
    '''
    def layer_constructor(self, *args, **kwargs):
        self.dw_conv = DWConv2d(*args, **kwargs)
        self.pw_conv = Conv2d(self.dw_conv, args[1], 1, **kwargs)
        return _layer_wrapper(self.dw_conv, self.pw_conv)
