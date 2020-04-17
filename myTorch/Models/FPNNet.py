try:
    from myTorch import nn, Models, Cat
    from myTorch.Models import EncodeDecodeBase
except ImportError:
    from .. import nn
    from .. import Models
    from . import EncodeDecodeBase
import torch


class _NameSetter:

    def __init__(self, name):
        self.name = name

    def __get__(self, obj, obj_type):
        return obj._level_prefix % (obj.i, self.name)


class FPNBuilder(EncodeDecodeBase):
    anti_alias_name = _NameSetter('anti_alias')

    def __init__(self, *args, top_down_channels=256, **kwargs):
        self.top_down_channels = top_down_channels
        super(FPNBuilder, self).__init__(*args, **kwargs)

    def construct_top_down_path(self, in_channels):
        self.bottle_neck_consolidator = nn.Conv2d(
                in_channels, self.top_down_channels, 1
        )
        self.bottle_neck_antialias = nn.Sequential(
                nn.Conv2d(self.top_down_channels, self.top_down_channels//2, 3),
                nn.Conv2d(
                    self.top_down_channels//2, self.top_down_channels//2, 3))

        iterable_layers = list(self.iterable_layers())

        self.number_of_scales = len(iterable_layers)

        layers_to_cat = [self.bottle_neck_antialias]

        for self.i, addable_layer in iterable_layers:
            in_channels = in_channels//2

            up_layer = torch.nn.Upsample(
                scale_factor=2, mode='bilinear', align_corners=True)
            setattr(self, self.upsample_name, up_layer)

            conv_layer = nn.Conv2d(
                in_channels, self.top_down_channels, 1)
            setattr(self, self.conv_name, conv_layer)
            anti_alias = nn.Sequential(
                nn.Conv2d(self.top_down_channels, self.top_down_channels//2, 3),
                nn.Conv2d(
                    self.top_down_channels//2, self.top_down_channels//2, 3))
            setattr(self, self.anti_alias_name, anti_alias)

            layers_to_cat.append(anti_alias)

        self.cat_layer = Cat(*layers_to_cat)

        return self.cat_layer

    def decoder_forward(self, x):
        iterable_layers = enumerate(reversed(self.skip_layers))

        x = self.bottle_neck_consolidator(x)
        anti_aliased_bottleneck = self.bottle_neck_antialias(x)
        upscaled_bottleneck = torch.nn.functional.upsample(
            anti_aliased_bottleneck, scale_factor=2**self.number_of_scales,
            mode='bilinear', align_corners=False)

        self.feature_pyramid_maps = [upscaled_bottleneck]
        for self.i, addable_layer in iterable_layers:
            x = getattr(self, self.upsample_name)(x)
            x = x + getattr(self, self.conv_name)(addable_layer)
            anti_aliased = getattr(self, self.anti_alias_name)(x)
            self.feature_pyramid_maps.append(
                torch.nn.functional.upsample(
                    anti_aliased, mode='bilinear', align_corners=False,
                    scale_factor=2**(self.number_of_scales-1-self.i)
                )
            )

        return self.cat_layer(*self.feature_pyramid_maps)


class ResnetFPN(FPNBuilder):
    output_activation = None

    def __init__(
        self, in_channels, num_classes, kernel_size=3,
            layers=[3, 3, 5, 2],
            base_channels=64,
            output_activation=None, **kwargs):
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

        super(ResnetFPN, self).__init__(
            encoder, num_classes, kernel_size, bottleneck=ResBottle, **kwargs)
        if output_activation is not None:
            self.output_activation = output_activation


class VGGFPN(FPNBuilder):
    output_activation = None

    def __init__(
            self, in_channels, num_classes, kernel_size, n_pool=4,
            base_channels=64, output_activation=None, layer_repeats=1, **kwargs):
        """__init__ Build a VGG FPN model

        https://arxiv.org/abs/1505.04597
        The model however uses instance normalization before each activation
        and uses Average pooling over max pooling.

        Args:
            in_channels ([type]): [description]
            num_classes ([type]): [description]
            kernel_size ([type]): [description]
            n_pool (int, optional): [description]. Defaults to 4.
            base_channels (int, optional): [description]. Defaults to 64.
            output_activation ([type], optional): [description]. Defaults to
                None.
        """
        layers = [layer_repeats]*n_pool
        encoder = Models.VGGNet(
            in_channels, base_channels, kernel_size,
            layers=layers)
        super(VGGFPN, self).__init__(encoder, num_classes, kernel_size, **kwargs)
        if output_activation is not None:
            self.output_activation = output_activation

class SEVGGFPN(FPNBuilder):
    decoder_constructor = nn.SEVGGBlock

    def __init__(
        self, in_channels, num_classes, kernel_size, n_pool=4,
        base_channels=64, output_activation=None, layer_repeats=1, **kwargs):
        """__init__ Build a SE FPN model

        https://arxiv.org/abs/1505.04597
        The model however uses instance normalization before each activation
        and uses Average pooling over max pooling.

        Args:
            in_channels ([type]): [description]
            num_classes ([type]): [description]
            kernel_size ([type]): [description]
            n_pool (int, optional): [description]. Defaults to 4.
            base_channels (int, optional): [description]. Defaults to 64.
            output_activation ([type], optional): [description]. Defaults to
                None.
        """
        layers = [layer_repeats]*n_pool
        encoder = Models.SEVGGNet(
            in_channels, base_channels, kernel_size,
            layers=layers)
        super(SEVGGFPN, self).__init__(encoder, num_classes, kernel_size, **kwargs)
        if output_activation is not None:
            self.output_activation = output_activation


class CSEVGGFPN(FPNBuilder):
    decoder_constructor = nn.CSEVGGBlock

    def __init__(
        self, in_channels, num_classes, kernel_size, n_pool=4,
        base_channels=64, output_activation=None, layer_repeats=1, **kwargs):
        """__init__ Build a CSEVGGFPN model

        https://arxiv.org/abs/1505.04597
        The model however uses instance normalization before each activation
        and uses Average pooling over max pooling.

        Args:
            in_channels ([type]): [description]
            num_classes ([type]): [description]
            kernel_size ([type]): [description]
            n_pool (int, optional): [description]. Defaults to 4.
            base_channels (int, optional): [description]. Defaults to 64.
            output_activation ([type], optional): [description]. Defaults to
                None.
        """
        layers = [layer_repeats]*n_pool
        encoder = Models.CSEVGGNet(
            in_channels, base_channels, kernel_size,
            layers=layers)
        super(CSEVGGFPN, self).__init__(encoder, num_classes, kernel_size, **kwargs)
        if output_activation is not None:
            self.output_activation = output_activation


class CSESEVGGFPN(FPNBuilder):
    decoder_constructor = nn.CSESEVGGBlock

    def __init__(
        self, in_channels, num_classes, kernel_size, n_pool=4,
        base_channels=64, output_activation=None, layer_repeats=1, **kwargs):
        """__init__ Build a CSESEVGGFPN FPN model

        https://arxiv.org/abs/1505.04597
        The model however uses instance normalization before each activation
        and uses Average pooling over max pooling.

        Args:
            in_channels ([type]): [description]
            num_classes ([type]): [description]
            kernel_size ([type]): [description]
            n_pool (int, optional): [description]. Defaults to 4.
            base_channels (int, optional): [description]. Defaults to 64.
            output_activation ([type], optional): [description]. Defaults to
                None.
        """
        layers = [layer_repeats]*n_pool
        encoder = Models.CSESEVGGNet(
            in_channels, base_channels, kernel_size,
            layers=layers)
        super(CSESEVGGFPN, self).__init__(encoder, num_classes, kernel_size, **kwargs)
        if output_activation is not None:
            self.output_activation = output_activation


if __name__ == "__main__":
    image = torch.zeros(1, 3, 256, 256)
    encoder = Models.ResNet(
            image, 1, 3,
            layers=[2, 3, 5, 2])
    model = ResnetFPN(3, 1, 3)
    output = model(image)
    print('done')
