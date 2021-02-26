import torch as _torch
from pathlib import Path as _Path
import numpy as _np
from collections import OrderedDict
try:
    from myTorch.Utils import (
        Isinstance, IfNoneReturnDefault)
    from myTorch.Data.Transforms import (
        TransformSequence
    )
except ImportError:
    from ..Utils import (
        Isinstance, IfNoneReturnDefault)
    from .Transforms import (
        TransformSequence)


class DictDataset(_torch.utils.data.Dataset):

    default_names = ['input', 'target']

    def __init__(
        self, folder_or_list, auto_gpu=True, transform=None, suffix='*.npz',
        input_name=None, target_name=None, RGB=False
    ):

        """__init__

        class for handling datasets containing dictionaries

        Args:
            folder_or_list ([str, Path, List, Tuple]):
                either a folder containing all the desired file or
                a list of files
            auto_gpu (bool, optional): if true assign
                data to the gpu by default. Defaults to True.
            transform ([function, list], optional):
                either a list of transforms or a function.
                if a list of tranforms then they are compiled to a
                single function. Defaults to None.
        """
        self.RGB = RGB
        self.input_name = input_name if input_name is not None \
            else self.default_names[0]
        # self.target_name = target_name if target_name is not None \
        #     else self.default_names[1]
        self.target_name = target_name

        self.suffix = suffix

        if Isinstance(_Path, str)(folder_or_list):
            self.mainfolder = _Path(folder_or_list)
            self.file_paths = list(
                self.mainfolder.glob(self.suffix)
            )
        else:
            assert Isinstance(tuple, list)(folder_or_list)
            self.file_paths = folder_or_list

        self.transform = IfNoneReturnDefault([])(transform)

        self.transform = TransformSequence(
            *self.transform, auto_gpu=auto_gpu)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        self.idx = idx
        if _torch.is_tensor(idx):
            idx = idx.to_list()
        file_path = self.file_paths[idx]
        file_dict = self.load_from_file(file_path)

        sample_data = self.grab_from_dict(file_dict)

        if self.transform:
            sample_data = self.transform(sample_data)
        
        # sample_data["filepath"] = [self.file_paths[idx], ]

        return sample_data

    def load_from_file(self, file_path):
        if _Path(file_path).suffix == ".npy":
            return _np.load(file_path, allow_pickle=True).item()
        if _Path(file_path).suffix == ".npz":
            return _np.load(file_path)

    def grab_from_dict(self, file_dict):
        data = OrderedDict()
        data[self.input_name] = self.parse_tensor(file_dict[self.input_name])
        if self.target_name is not None:
            data[self.target_name] = \
                self.parse_tensor(file_dict[self.target_name])
        miscallaneous_keys = [
            key for key in file_dict.keys() if
            key not in [self.input_name, self.target_name]]
        for key in miscallaneous_keys:
            data[key] = self.parse_tensor(file_dict[key])
        return data

    def parse_tensor(self, tensor):

        tensor = _torch.Tensor(tensor)

        if self.RGB:
            if len(tensor.shape) == 3:
                return _torch.Tensor(tensor).permute(2, 0, 1)
            elif len(tensor.shape) == 2:
                return _torch.Tensor(tensor).unsqueeze(0)
            else:
                raise Exception
        else:
            if len(tensor.shape) == 2:
                return _torch.Tensor(tensor).unsqueeze(0)
        return tensor


class DictDatasetVolume(DictDataset):

    def __init__(
        self, folder_or_list, auto_gpu=True, transform=None, suffix='*.npy',
        input_name=None, target_name=None,
    ):
        super().__init__(folder_or_list, auto_gpu=auto_gpu, transform=transform, suffix=suffix,
        input_name=input_name, target_name=target_name, RGB=False)

    def parse_tensor(self, tensor):

        tensor = _torch.Tensor(tensor)

        if len(tensor.shape) == 3:
            return _torch.Tensor(tensor).unsqueeze(0)

        return tensor


class GraphDataset(DictDataset):
    def parse_tensor(self, tensor):
        return _torch.tensor(tensor)

if __name__ == "__main__":
    fake_files = ['1', '2']
    dataset_1 = DictDataset(fake_files, transform=[1, 2, 3])
    dataset_2 = DictDataset(fake_files, transform=[4, 5, 6])
    print('done')
