from sklearn import model_selection
from . import DictDataset
import torch
import numpy as np
from typing import Iterable


class TrainTestSplit:

    def __init__(self, iterable, **dataset_kwargs):
        dataset_kwargs['auto_gpu'] = dataset_kwargs.get('auto_gpu', True)
        self.iterable = iterable
        self.dataset_kwargs = dataset_kwargs
        self.is_split = False

    def __call__(self, **loader_kwargs):
        assert self.is_split, 'must run {self}.split(**kwargs)'
        test_dataset_kwargs = self.dataset_kwargs.copy()
        test_dataset_kwargs['transform'] = None
        # build the train and test dataset
        train_dataset = DictDataset(
            self.XY_train, **self.dataset_kwargs)
        test_dataset = DictDataset(self.XY_test, **test_dataset_kwargs)
        #
        test_loader_kwargs = loader_kwargs.copy()
        test_loader_kwargs['shuffle'] = False
        # using the train and test dataset build dataloaders
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, **loader_kwargs)
        test_dataloader = torch.utils.data.DataLoader(
            test_dataset, **test_loader_kwargs)
        return train_dataloader, test_dataloader

    def split(self, **split_kwargs):
        '''
        Args:
            test_size : float, int or None, optional (default=None)
                If float, should be between 0.0 and 1.0 and represent the proportion
                of the dataset to include in the test split. If int, represents the
                absolute number of test samples.

            train_size : float, int, or None, (default=None)
                If float, should be between 0.0 and 1.0 and represent the
                proportion of the dataset to include in the train split. If
                int, represents the absolute number of train samples.

            random_state : int, RandomState instance or None, optional (default=None)
                If int, random_state is the seed used by the random number generator;
                If RandomState instance, random_state is the random number generator;
                If None, the random number generator is the RandomState instance used
                by `np.random`.

            shuffle : boolean, optional (default=True)
                Whether or not to shuffle the data before splitting. If shuffle=False
                then stratify must be None.

            stratify : array-like or None (default=None)
                If not None, data is split in a stratified fashion, using this as
                the class labels.
        '''
        self.XY_train, self.XY_test = model_selection.train_test_split(
            self.iterable, **split_kwargs)
        self.is_split = True


class TrainValidationTestSplit:

    def __init__(self, iterable: Iterable, auto_gpu=True, transform=None,
                 suffix='*.npy', input_name=None, target_name=None, dataset_method=DictDataset,
                 **kwargs):
        """__init__

        Split an iterable into a train validation test split

        Args:
            iterable (Iterable): some iterable to divide into a train
                validation, test split
            auto_gpu (bool, optional): specify whether or not to automatically
                place the data onto the gpu Defaults to True.
            transform ([Iterable], optional): an iterable of transformations.
                Defaults to None.
            suffix (str, optional): if rather than an iterable a string
                directed to a folder is provided an iterable is generated from
                the files with the designated suffix. Defaults to '*.npy'.
            input_name ([type], optional): the items within the iterable are
                assumed to be dictionaries. input_name is the dictioanry key
                pointing to the model input. Defaults to None.
            target_name ([type], optional): the items within the iterable are
                assumed to be dictionaries. target_name is the dictioanry key
                pointing to the desired the model input. Defaults to None.

        Returns:
            [type]: [description]
        """
        self.iterable = iterable
        self.dataset_kwargs = dict(
            auto_gpu=auto_gpu, transform=transform, suffix=suffix,
            input_name=input_name, target_name=target_name, **kwargs
        )
        self.is_split = False
        self.dataset_method = dataset_method

    def __call__(self, **loader_kwargs):
        '''
        Args:
            batch_size (int, optional): how many samples per batch to load
                (default: ``1``).
            shuffle (bool, optional): set to ``True`` to have the data
                reshuffled at every epoch (default: ``False``).
            sampler (Sampler, optional): defines the strategy to draw samples
                from the dataset. If specified, :attr:`shuffle` must be
                False``.
            batch_sampler (Sampler, optional): like :attr:`sampler`, but
                returns a batch of indices at a time. Mutually exclusive with
                :attr:`batch_size`, :attr:`shuffle`, :attr:`sampler`, and
                :attr:`drop_last`.
            num_workers (int, optional): how many subprocesses to use for data
                loading. ``0`` means that the data will be loaded in the main
                process. (default: ``0``)
            collate_fn (callable, optional): merges a list of samples to form a
                mini-batch of Tensor(s).  Used when using batched loading from
                a map-style dataset.
            pin_memory (bool, optional): If ``True``, the data loader will
                copy Tensors into CUDA pinned memory before returning them.
                If your data elements are a custom type, or your :attr:
                collate_fn` returns a batch that is a custom type, see the
                example below.
            drop_last (bool, optional): set to ``True`` to drop the last
                incomplete batch, if the dataset size is not divisible by the
                batch size. If ``False`` and the size of dataset is not
                divisible by the batch size, then the last batch
                will be smaller. (default: ``False``)
            timeout (numeric, optional): if positive, the timeout value for
                collecting a batch from workers. Should always be non-negative.
                (default: ``0``)
            worker_init_fn (callable, optional): If not ``None``, this will be
                called on each worker subprocess with the worker id
                (an int in ``[0, num_workers - 1]``) as
                input, after seeding and before data loading.
                (default: ``None``)
        '''
        assert self.split, 'must run {self}.split(**kwargs)'
        test_dataset_kwargs = self.dataset_kwargs.copy()
        test_dataset_kwargs['transform'] = None
        # build the train valid test dataset
        train_dataset = self.dataset_method(self.XY_train, **self.dataset_kwargs)
        valid_dataset = self.dataset_method(self.XY_valid, **self.dataset_kwargs)
        test_dataset = self.dataset_method(self.XY_test, **test_dataset_kwargs)
        # generate the input to the train/valid and test loaders
        test_loader_kwargs = loader_kwargs.copy()
        test_loader_kwargs['shuffle'] = False
        # build the dataloaders
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, **loader_kwargs)
        valid_dataloader = torch.utils.data.DataLoader(
            valid_dataset, **loader_kwargs)
        test_dataloader = torch.utils.data.DataLoader(
            test_dataset, **test_loader_kwargs)
        return train_dataloader, valid_dataloader, test_dataloader

    def split(
            self, train_size, valid_size, test_size, **kwargs):
        '''
        Args:
            test_size : float, int or None, optional (default=None)
                If float, should be between 0.0 and 1.0 and represent the proportion
                of the dataset to include in the test split. If int, represents the
                absolute number of test samples.

            valid_size : float, int or None, optional (default=None)
                If float, should be between 0.0 and 1.0 and represent the proportion
                of the dataset to include in the test and train split. If int, represents the
                absolute number of test samples.

            train_size : float, int, or None, (default=None)
                If float, should be between 0.0 and 1.0 and represent the
                proportion of the dataset to include in the train split. If
                int, represents the absolute number of train samples.

            random_state : int, RandomState instance or None, optional (default=None)
                If int, random_state is the seed used by the random number generator;
                If RandomState instance, random_state is the random number generator;
                If None, the random number generator is the RandomState instance used
                by `np.random`.

            shuffle : boolean, optional (default=True)
                Whether or not to shuffle the data before splitting. If shuffle=False
                then stratify must be None.

            stratify : array-like or None (default=None)
                If not None, data is split in a stratified fashion, using this as
                the class labels.
        '''
        assert np.isclose(train_size + valid_size + test_size, 1.0), \
            "train_size + test_size + valid_size must be 1"
        # stage 1 split to a train/valid and test split
        temp, self.XY_test = model_selection.train_test_split(
            self.iterable, test_size=test_size, **kwargs)
        # stage 2 split the train/valid dataset
        percentage_train_valid = train_size + valid_size
        valid_size = valid_size/percentage_train_valid
        self.XY_train, self.XY_valid = model_selection.train_test_split(
            temp,  test_size=valid_size, **kwargs)
        # data has been split can now call
        self.is_split = True


class KFoldCrossTrainTestSplit(TrainValidationTestSplit):

    def __call__(self, **loader_kwargs):
            '''
            Args:
                batch_size (int, optional): how many samples per batch to load
                    (default: ``1``).
                shuffle (bool, optional): set to ``True`` to have the data
                    reshuffled at every epoch (default: ``False``).
                sampler (Sampler, optional): defines the strategy to draw samples
                    from the dataset. If specified, :attr:`shuffle` must be
                    False``.
                batch_sampler (Sampler, optional): like :attr:`sampler`, but
                    returns a batch of indices at a time. Mutually exclusive with
                    :attr:`batch_size`, :attr:`shuffle`, :attr:`sampler`, and
                    :attr:`drop_last`.
                num_workers (int, optional): how many subprocesses to use for data
                    loading. ``0`` means that the data will be loaded in the main
                    process. (default: ``0``)
                collate_fn (callable, optional): merges a list of samples to form a
                    mini-batch of Tensor(s).  Used when using batched loading from
                    a map-style dataset.
                pin_memory (bool, optional): If ``True``, the data loader will
                    copy Tensors into CUDA pinned memory before returning them.
                    If your data elements are a custom type, or your :attr:
                    collate_fn` returns a batch that is a custom type, see the
                    example below.
                drop_last (bool, optional): set to ``True`` to drop the last
                    incomplete batch, if the dataset size is not divisible by the
                    batch size. If ``False`` and the size of dataset is not
                    divisible by the batch size, then the last batch
                    will be smaller. (default: ``False``)
                timeout (numeric, optional): if positive, the timeout value for
                    collecting a batch from workers. Should always be non-negative.
                    (default: ``0``)
                worker_init_fn (callable, optional): If not ``None``, this will be
                    called on each worker subprocess with the worker id
                    (an int in ``[0, num_workers - 1]``) as
                    input, after seeding and before data loading.
                    (default: ``None``)
            '''
            assert self.split, 'must run {self}.split(**kwargs)'

            self.fold_loaders = []

            for train, valid, test in self.folds:
                test_dataset_kwargs = self.dataset_kwargs.copy()
                test_dataset_kwargs['transform'] = None
                # build the train valid test dataset
                train_dataset = self.dataset_method(train, **self.dataset_kwargs)
                valid_dataset = self.dataset_method(valid, **self.dataset_kwargs)
                test_dataset = self.dataset_method(test, **test_dataset_kwargs)
                # generate the input to the train/valid and test loaders
                test_loader_kwargs = loader_kwargs.copy()
                test_loader_kwargs['shuffle'] = False
                # build the dataloaders
                train_dataloader = torch.utils.data.DataLoader(
                    train_dataset, **loader_kwargs)
                valid_dataloader = torch.utils.data.DataLoader(
                    valid_dataset, **loader_kwargs)
                test_dataloader = torch.utils.data.DataLoader(
                    test_dataset, **test_loader_kwargs)
                self.fold_loaders.append(
                    (train_dataloader, valid_dataloader, test_dataloader)
                )

            return self.fold_loaders

    def split(
            self, valid_size, n_splits=5, **kwargs):
        '''
        Args:
            test_size : float, int or None, optional (default=None)
                If float, should be between 0.0 and 1.0 and represent the proportion
                of the dataset to include in the test split. If int, represents the
                absolute number of test samples.

            valid_size : float, int or None, optional (default=None)
                If float, should be between 0.0 and 1.0 and represent the proportion
                of the dataset to include in the test and train split. If int, represents the
                absolute number of test samples.

            random_state : int, RandomState instance or None, optional (default=None)
                If int, random_state is the seed used by the random number generator;
                If RandomState instance, random_state is the random number generator;
                If None, the random number generator is the RandomState instance used
                by `np.random`.

            shuffle : boolean, optional (default=True)
                Whether or not to shuffle the data before splitting. If shuffle=False
                then stratify must be None.
        '''

        # stage 1 split the data into KFOLDS
        kf = model_selection.KFold(
            n_splits=n_splits, shuffle=kwargs.get('shuffle', False),
            random_state=kwargs.get('random_state', None))
        kf.split(self.iterable)

        self.folds = []

        # stage 2 loop through the data
        for train_index, test_index in kf.split(self.iterable):
            XY_test = [self.iterable[i] for i in test_index]
            temp = [self.iterable[i] for i in train_index]
            XY_train, XY_valid = model_selection.train_test_split(
                temp,  test_size=valid_size, **kwargs)
            self.folds.append(
                (XY_train, XY_valid, XY_test)
            )
        # data has been split can now call
        self.is_split = True


__all__ = [
    'TrainTestSplit', 'TrainValidationTestSplit'
]
