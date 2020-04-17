import torch as _torch


class LayerPartial:
    def __init__(self, method, **kwargs):
        self.method = method
        self.kwargs = kwargs

    def __call__(self, *args, **kwargs):
        self.kwargs.update(kwargs)
        return self.method(*args, **self.kwargs)


class Layer(_torch.nn.Module):

    def __init__(self, *args, **kwargs):
        super(Layer, self).__init__()
        self.parse_arguments(*args, **kwargs)

    def forward(self, x, **kwargs):
        x = self.pre_forward(x, **kwargs)
        x = self.main_forward(x, **kwargs)
        x = self.post_forward(x, **kwargs)
        return x

    def pre_forward(self, x, **kwargs):
        return x

    def main_forward(self, x, **kwargs):
        return x

    def post_forward(self, x, **kwargs):
        return x

    def parse_arguments(self, *args, **kwargs):
        args = list(args)
        if args:
            if issubclass(args[0].__class__, Layer):
                args[0] = args[0].out_channels
            if issubclass(args[0].__class__, _torch.Tensor):
                args[0] = args[0].shape[1]
        if not hasattr(self, 'args'):
            self.args = args
            self.kwargs = kwargs
