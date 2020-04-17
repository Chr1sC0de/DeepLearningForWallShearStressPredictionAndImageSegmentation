try:
    from myTorch.nn import Layer
except ImportError:
    from .base import Layer
import torch as _torch


class _periodic_pad_controller:
    """ controls the way in which padding is performed

    [extended_summary]

    Returns:
        [type]: [description]
    """
    def __init__(self, padding_tuple, switches):
        self.padding_tuple = padding_tuple
        self.switches = switches

    def __call__(self, x):
        if sum(self.switches) > 1:
            if sum(self.switches[0:2]) == 2:
                x = self.periodic_pad(x, self.padding_tuple[0], direction='c')
            if sum(self.switches[2:]) == 2:
                x = self.periodic_pad(x, self.padding_tuple[-1], direction='r')
        else:
            all_directions = ['left', 'right', 'top', 'bottom']
            for direction, p, switch in zip(all_directions, self.padding_tuple, self.switches):
                if switch:
                    x = getattr(
                        self, f'pad_{direction}'
                    )(x, p)
        return x

    def periodic_pad(self, x, p, direction='r'):
        if p == 0:
            return x

        if direction == 'r':
            dim = 2
            x = _torch.cat(
                [x[:, :, -p:, :], x, x[:, :, :p, :]], dim=dim
            )

        if direction == 'c':
            dim = 3
            x = _torch.cat(
                [x[:, :, :, -p:], x, x[:, :, :, :p]], dim=dim
            )

        return x

    def pad_left(self, x, p):
        return _torch.cat([x[:, :, -p:, :], x], dim=2)

    def pad_right(self, x, p):
        return _torch.cat([x, x[:, :, :p, :]], dim=2)

    def pad_top(self, x, p):
        return _torch.cat([x[:, :, :, -p:], x], dim=3)

    def pad_bottom(self, x, p):
        return _torch.cat([x[:, :, :, p:], x], dim=3)


def calculate_padding_1D(
    in_size,
    out_size,
    kernel_size,
    stride,
    dilation
):
    i, o, k = in_size, out_size, kernel_size
    s, d = stride, dilation

    a = o - 1
    b = k - 1

    p = (s*a-i+d*b+1)//2

    return p


class _DirectionMethod:
    def __init__(self, padding_tuple):
        self.padding_tuple = padding_tuple

    def __call__(self, myClass, *args):
        obj = myClass(*args)
        obj._padding_switches = self.padding_tuple
        return obj


class _SamePadding(Layer):
    _padding_switches = (1, 1, 1, 1)
    _padding_method = None
    _paddingargs = ()

    def __init__(
        self,
        kernel_size,
        stride,
        dilation
    ):
        super(_SamePadding, self).__init__()
        self.padding_tuple = (kernel_size, stride, dilation)
        self.constructed = False

    def forward(self, x):

        self._construct_padder(x)
        self.constructed = True
        self.out_channels = x.shape[1]
        self.in_channels = x.shape[1]

        return self.padder(x)

    def _construct_padder(self, x):
        # if not self.constructed:
        row_pad = calculate_padding_1D(
            x.shape[2], x.shape[2], *self.padding_tuple)
        col_pad = calculate_padding_1D(
            x.shape[3], x.shape[3], *self.padding_tuple)

        lrtb_padding = [row_pad, row_pad, col_pad, col_pad]

        padding_sizes = [
            lrtb_padding[0]*self._padding_switches[0],
            lrtb_padding[1]*self._padding_switches[1],
            lrtb_padding[2]*self._padding_switches[2],
            lrtb_padding[3]*self._padding_switches[3]
        ]

        if isinstance(self._padding_method, str):
            self.padder = getattr(
                _torch.nn, self._padding_method
                )(padding_sizes)
        else:
            self.padder = self._padding_method(
                padding_sizes, *self._paddingargs
                )


_SamePadding.left = classmethod(_DirectionMethod((1, 0, 0, 0)))
_SamePadding.right = classmethod(_DirectionMethod((0, 1, 0, 0)))
_SamePadding.top = classmethod(_DirectionMethod((0, 0, 1, 0)))
_SamePadding.bottom = classmethod(_DirectionMethod((0, 0, 0, 1)))
_SamePadding.left_right = classmethod(_DirectionMethod((1, 1, 0, 0)))
_SamePadding.top_bottom = classmethod(_DirectionMethod((0, 0, 1, 1)))


class ZeroPad2d(_SamePadding):
    _padding_method = 'ZeroPad2d'


class ConstantPad2d(_SamePadding):
    _padding_method = 'ConstantPad2d'
    _paddingargs = (1)


class ReplicationPad2d(_SamePadding):
    _padding_method = 'ReplicationPad2d'


class PeriodicPad2d(_SamePadding):

    # note does not support padding in only a single direction

    def _padding_method(self, padding_sizes):
        return _periodic_pad_controller( 
            padding_sizes, self._padding_switches
        )


class PeriodicReplication2d(_SamePadding):

    def _padding_method(self, padding_sizes):

        row_padder = PeriodicPad2d.top_bottom(*self.padding_tuple)
        column_padder = ReplicationPad2d.left_right(*self.padding_tuple)

        def output_method(x):
            x = row_padder(x)
            x = column_padder(x)
            return x

        return output_method
