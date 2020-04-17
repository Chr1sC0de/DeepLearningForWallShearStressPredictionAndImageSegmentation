try:
    from .. import nn as _nn
except ImportError:
    from myTorch import nn as _nn


class _MergerBase(_nn.Layer):

    def __init__(self, *args, **kwargs):
        super(_MergerBase, self).__init__(*args, **kwargs)
        self.layer_registry = []
        self._skip_layers = []
        self.track_skip = kwargs.get('track_skip', False)

    def custom_register_layers(self, *sequence_list):
        for i, sequence in enumerate(sequence_list):

            if not issubclass(sequence.__class__, _nn.Sequential):
                for j, layer in enumerate(sequence):
                    setattr(self, 'level_%d_layer_%d' % (i, j), layer)
                    self.layer_registry.append(layer)

                current_layer = layer
            else:
                setattr(self, 'level_%d' % i, layer)
                self.layer_registry.append(layer)
                current_layer = layer

            current_layer.tail_layer = True

            if i < (len(sequence_list)-1):
                pool_layer = self.connector_constructor(current_layer)
                setattr(self, 'pool_%d' % i, pool_layer)
                self.layer_registry.append(pool_layer)

        self.out_channels = current_layer.out_channels

    def connector_constructor(self, input_layer):
        return _nn.AvgPool2d(input_layer, 2)

    def main_forward(self, x, **kwargs):
        if self.track_skip:
            self._skip_layers = []

        for i, layer in enumerate(self.layer_registry):
            x = layer(x)
            if self.track_skip:
                if hasattr(layer, 'tail_layer'):
                    self._skip_layers.append(x)
        return x

    @property
    def concatenable_layers(self):
        concat_layers = []
        for i, level in enumerate(self.layer_registry):
            if hasattr(level, 'tail_layer'):
                concat_layers.append(level)
        return concat_layers

    @property
    def skip_layers(self):
        return self._skip_layers


class MergedFromSequence(_MergerBase):
    def __init__(self, *sequence_list, track_skip=False, **kwargs):
        super(MergedFromSequence, self).__init__(**kwargs)
        self.custom_register_layers(*sequence_list)


class MergedFromPattern(_MergerBase):
    layer_constructor = None
    layers = [1, 1, 1, 1]
    scale_factor = 2

    def __init__(self, *args, layers=None, scale_factor=None, **kwargs):
        """__init__

        arg: (input, base_channels, kernel_size)
        """
        super(MergedFromPattern, self).__init__(*args, **kwargs)
        if layers is not None:
            self.layers = layers
        if scale_factor is not None:
            self.scale_factor = scale_factor
        self.parse_arguments(*args, *{})
        self.generate_sequence()
        self.custom_register_layers(*self.sequence_of_sequences)

    def generate_sequence(self):
        self.sequence_of_sequences = []
        in_channels, base_channels,  kernel_size = self.args

        for self.n_repeats in self.layers:

            self.sequence = []

            for self.i in range(self.n_repeats):

                if not self.sequence:
                    self.initialize_sequence()
                else:
                    self.append_sequence()

            self.sequence_of_sequences.append(self.sequence)

    def initialize_sequence(self):
        if self.sequence_of_sequences:
            in_channels = self.sequence_of_sequences[-1][-1].out_channels
            out_channels = int(in_channels*self.scale_factor)
            self.sequence.append(
                self.layer_constructor(
                    in_channels, out_channels, self.args[-1])
            )
        else:
            layer = self.layer_constructor(*self.args)
            layer.in_channels = self.args[0]
            self.sequence.append(layer)

    def append_sequence(self):
        in_channels = self.sequence[-1].out_channels
        kernel_size = self.args[-1]
        self.sequence.append(
            self.layer_constructor(
                in_channels, in_channels, kernel_size
            )
        )


class ResNet(MergedFromPattern):
    layer_constructor = _nn.ResidualBlock
    layers = [3, 3, 5, 2]

    def connector_constructor(self, input_layer):
        layer_dict = dict(stride=[2, 1], padding=[1, 'same'])
        connector = _nn.ResidualBlock(
            input_layer.out_channels, input_layer.out_channels, 3,
            layer_dict=layer_dict)
        return connector


class ResNetBottleNeck(MergedFromPattern):
    layer_constructor = _nn.BottleNeckResidualBlock
    layers = [3, 3, 5, 2]

    def connector_constructor(self, input_layer):
        connector = _nn.BottleNeckResidualBlock(
            input_layer.out_channels, input_layer.out_channels, 3,
            stride=2)
        return connector


class VGGNet(MergedFromPattern):
    layer_constructor = _nn.VGGBlock
    layers = [1, 1, 1, 1, 1]

    def connector_constructor(self, input_layer):
        return _nn.AvgPool2d(input_layer, 2)

class VGGNet3D(MergedFromPattern):
    layer_constructor = _nn.VGGBlock3d
    layers = [1, 1, 1]

    def connector_constructor(self, input_layer):
        return _nn.AvgPool3d(input_layer, 2)


class SEVGGNet(MergedFromPattern):
    layer_constructor = _nn.SEVGGBlock
    layers = [1, 1, 1, 1, 1]

    def connector_constructor(self, input_layer):
        return _nn.AvgPool2d(input_layer, 2)


class CSEVGGNet(MergedFromPattern):
    layer_constructor = _nn.CSEVGGBlock
    layers = [1, 1, 1, 1, 1]

    def connector_constructor(self, input_layer):
        return _nn.AvgPool2d(input_layer, 2)


class CSESEVGGNet(MergedFromPattern):
    layer_constructor = _nn.CSESEVGGBlock
    layers = [1, 1, 1, 1, 1]

    def connector_constructor(self, input_layer):
        return _nn.AvgPool2d(input_layer, 2)


class DilatedResnet(MergedFromSequence):

    def __init__(self, layers=[1, 1, 1], track_skip=False, **kwargs):
        ResNet()

    def connector_constructor(self, input_layer):
        layer_dict = dict(stride=[2, 1], padding=[1, 'same'])
        connector = _nn.ResidualBlock(
            input_layer.out_channels, input_layer.out_channels, 3,
            layer_dict=layer_dict)
        return connector


if __name__ == '__main__':

    import torch as _torch
    from torch import nn,optim
    from torch.utils.tensorboard import SummaryWriter
    filterSize = 3
    strides = 1
    dilationRate = 1
    from torch.utils.tensorboard import SummaryWriter

    writer = SummaryWriter()


    argsTuple = (filterSize, strides, dilationRate)

    XYZ = _torch.meshgrid(*[_torch.arange(0, 16)]*2)
    testObj = _torch.stack(XYZ, dim=0)
    testObj = testObj.unsqueeze(0).float()

    print(testObj.shape)

    conv_1 = ResNet(2, 32, 3, track_skip=True)
    x = conv_1(testObj)

    writer.add_graph(conv_1, testObj)
    writer.flush()
    writer.close()

    criterion = nn.L1Loss()
    optimizer = optim.SGD(conv_1.parameters(), lr=0.001, momentum=0.9)

    # the loop
    optimizer.zero_grad()
    outputs = conv_1(testObj)
    loss = criterion(outputs, testObj)
    loss.backward()
    optimizer.step()
    print('done')


















