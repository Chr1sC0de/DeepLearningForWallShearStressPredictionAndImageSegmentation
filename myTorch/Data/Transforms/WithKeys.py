import numpy as _np
from ... import _callback_decriptor
import torch

class FromKeys:
    """ Methods that modifies a dictionary based on its keys
    """
    pre_transform = _callback_decriptor('pre_transform')
    post_transform = _callback_decriptor('post_transform')

    def __init__(self, *args, **kwargs):
        """__init__

        Transform from a set of keys.

        """
        self.callback_list = kwargs.get('callbacks', [])

    def forward(self, sample_dict):
        NotImplementedError

    def __call__(self, sample_dict):
        self.pre_transform
        self.sample_dict = self.forward(sample_dict)
        self.post_transform
        return self.sample_dict


class RollTensor(FromKeys):

    def forward(self, sample_dict):
        key = next(iter(sample_dict.keys()))
        c, h, w = sample_dict[key].shape
        roll_val = _np.random.randint(h)
        for key in sample_dict.keys():
            sample_dict[key] = sample_dict[key].roll(roll_val, 1)
        return sample_dict


class FlipTensor(FromKeys):

    def forward(self, sample_dict):
        key = next(iter(sample_dict.keys()))

        for key in sample_dict.keys():
            if _np.random.randint(0,2):
                flip_dim = _np.random.randint(1,3)
                sample_dict[key] = torch.flip(
                    sample_dict[key], [flip_dim]
                )
        return sample_dict


class ToDtype(FromKeys):
    def __init__(self, dtype, *args, **kwargs):
        super(ToDtype, self).__init__(*args, **kwargs)
        self.dtype = dtype

    def forward(self, sample_dict):
        for key in sample_dict.keys():
            try:
                sample_dict[key] = sample_dict[key].type(self.dtype)
            except TypeError:
                print(f'could not convert {sample_dict[key]} to {self.dtype}')

        return sample_dict
