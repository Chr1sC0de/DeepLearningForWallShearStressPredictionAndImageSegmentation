import pathlib
import abc
import copy
import torch
import json
from . import nn
import numpy as np


class Cat(nn.Layer):

    def __init__(self, *args):
        # initialize concatenation layer from input layers
        super(Cat, self).__init__()

        self.in_channels = 0
        self.out_channels = 0

        if issubclass(args[0].__class__, torch.Tensor):
            self.__get_torch_channels(*args)
        else:
            self.__get_layer_channels(*args)

        self.in_channels = self.out_channels

    def forward(self, *args):
        return torch.cat(args, dim=1)

    def __get_torch_channels(self, *args):
        for arg in args:
            self.out_channels += args.shape[1] 

    def __get_layer_channels(self, *args):
        for arg in args:
            self.out_channels += arg.out_channels

    @classmethod
    def from_layer(cls, layer, *args, **kwargs):
        raise "input layers directly into the constructor"


def negative_to_positive_finder(
        data, window_scale=4, polyorder=3):

    pre_val = data[0]
    min_found = []
    for i, val in enumerate(data[1:]):
        if all(
                [pre_val < 0, val > 0]):
            min_found.append(i)
        pre_val = val

    return min_found


class Isinstance:
    def __init__(self, *args):
        self.instances = args

    def __call__(self, obj):
        for instance in self.instances:
            if isinstance(obj, instance):
                return True
        return False


class GlobFiles:

    def __init__(self, pattern):
        self.pattern = pattern

    def __call__(self, folder):
        return list(pathlib.Path(folder).glob(self.pattern))


_glob_numpy = GlobFiles("*.npy")
_glob_npz = GlobFiles("*.npz")


def glob_numpy(folder):
    return list(_glob_numpy(folder))

def glob_npz(folder):
    return list(_glob_npz(folder))


class DefaultReturn:
    def __init__(self, default):
        self.default = default

    def __call__(self, value):
        if self.condition(value):
            return self.default
        else:
            return value

    @abc.abstractmethod
    def condition(self, value):
        pass


class IfNoneReturnDefault(DefaultReturn):
    def condition(self, value):
        if value is None:
            return True
        return False


class ToDevice:
    def __init__(self, device):
        self.device = device

    def __call__(self, *args):
        output_args = []
        for arg in args:
            if isinstance(arg, (list, tuple)):
                output_args.append([item.to(self.device) for item in arg])
            else:
                output_args.append(arg.to(self.device))
        return output_args


class StateCacher(object):

    def __init__(self, in_memory, cache_dir=None):
        self.in_memory = in_memory
        self.cache_dir = cache_dir

        if self.cache_dir is None:
            import tempfile
            self.cache_dir = tempfile.gettempdir()
        else:
            if not pathlib.Path(self.cache_dir).is_dir():
                raise ValueError(f'{self.cache_dir} is not a valid directory.')
            else:
                self.cache_dir = pathlib.Path(self.cache_dir)

        self.cached = {}

    def store(self, key, state_dict):
        if self.in_memory:
            self.cached.update(
                {key: copy.deepcopy(state_dict)}
            )
        else:
            fn = self.cache_dir/f"state_{key}_{id(self)}.pt"
            self.cached.update({key: fn})
            torch.save(state_dict, fn)

    def retrieve(self, key):
        if key not in self.cached:
            raise KeyError(f"Target {key} was not in cached")

        if self.in_memory:
            return self.cached.get(key)
        else:
            fn = self.cached.get(key)
            if not pathlib.Path(fn).exists():
                raise RuntimeError(f'Failed to load state in {fn}\
                    . File does not exists anymore')
            state_dict = torch.load(
                fn, map_location=lambda storage, location: storage)
            return state_dict


class AttrDict(dict):

    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def json_to_attrdict(path):
    with open(path, 'r') as f:
        json_dict = json.load(f)
    output = AttrDict()
    output.update(json_dict)
    return output


def load_numpy_item(path):
    return np.load(path, allow_pickle=True).item()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())



