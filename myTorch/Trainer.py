import torch as _torch
from .Callbacks import Stdout
from functools import wraps as _wraps
from .optim import LRFinder, LRFinderCFD
from . import nn

stdout_config = {

}


class _callback_decriptor:
    def __init__(self, name):
        self.name = name

    def __get__(self, model, modelType):
        if model.callback_list:
            for callback in model.callback_list:
                getattr(callback, self.name)(model)


_stdout_kwargs = ['print_on_batch', 'print_epoch', 'print_cycle',
                  'ma_length', 'print_loss', 'print_accuracy',
                  'file_path', 'loss_format',
                  'metric_dict', 'std_to_file']


class AssignAndCleanCallbacks:

    def __init__(self, **kwargs):
        self.default_kwargs = kwargs

    def __call__(self, function):
        @_wraps(function)
        def wrapper(obj, *args, **kwargs):
            self.default_kwargs.update(kwargs)

            for key in [key for key in self.default_kwargs.keys()
                        if key not in _stdout_kwargs]:
                self.default_kwargs.pop(key)

            obj.callback_list.extend(
                kwargs.get('callbacks', [])
            )
            is_stdout_avail = False
            for callback in obj.callback_list:
                if issubclass(callback.__class__, Stdout):
                    is_stdout_avail = True
            if not is_stdout_avail:
                obj.callback_list.extend(
                    [Stdout(**self.default_kwargs)]
                )
            function(obj, *args, **kwargs)
            obj.callback_list = []
        return wrapper


class Trainer:
    auto_assign = True

    pre_loop = _callback_decriptor('pre_loop')
    pre_cycle = _callback_decriptor('pre_cycle')
    pre_epoch = _callback_decriptor('pre_epoch')
    pre_batch = _callback_decriptor('pre_batch')
    post_batch = _callback_decriptor('post_batch')
    post_epoch = _callback_decriptor('post_epoch')
    post_cycle = _callback_decriptor('post_cycle')
    post_loop = _callback_decriptor('post_loop')

    def __init__(self, model, optimizer, criterion,
                 model_initializer=None, x_key=None, y_key=None):
        """__init__

        An environment for training and testing supervised models

        Args:
            model ([nn.Module]): a pytorch module
            optimizer ([nn.Module]): a pytorch
                optimizer function, must have a step
                and zero_grad and step method
            loss ([nn.functional]):
                function which calculates the loss
        """
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.callback_list = []
        self.pre_extractor_hooks = []
        if self.auto_assign:
            self.place_on_gpu_if_available()
        if model_initializer is not None:
            self.model = model_initializer(self.model)
        self.x_key = x_key
        self.y_key = y_key

    def lr_find(
            self, train_loader, plot=False, use_recommended=True, **kwargs):
        if not hasattr(self, 'lr_finder'):
            self.lr_finder = LRFinder(
                self.model, self.optimizer, self.criterion,
            )
        self.lr_finder.range_test(train_loader, **kwargs)
        if plot:
            self.lr_finder.plot()
        if use_recommended:
            min_lr, max_lr = self.lr_finder.recommend_lr()
            self.optimizer.param_groups[0]['lr'] = min_lr

    def place_on_gpu_if_available(self):
        self.device = _torch.device(
            "cuda:0" if _torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)
        for parameter in self.model.parameters():
            parameter.to(self.device)
        print(f"the model is assigned to {self.device}")

    @AssignAndCleanCallbacks(
        **dict(print_on_batch=50, print_epoch=True, print_cycle=True,
               ma_length=20, print_loss=True, print_accuracy=True,
               loss_format='%0.3f', metric_dict=None, std_to_file=None))
    def fit_on_loader(
        self, train_dataloader, cycles=[4, ], valid_dataloader=None,
        callbacks=None, **kwargs
    ):
        self.break_batch_loop = False
        self.break_epoch_loop = False
        self.break_cycle_loop = False
        self.n_cycles = len(cycles)
        # self.model.train()
        self.valid_dataloader = valid_dataloader
        self.train_dataloader = train_dataloader
        self.main_dataloader = train_dataloader
        # self.pre_loop
        [callback.pre_loop(self) for callback in self.callback_list]
        for self.i_cycle, self.n_epochs in enumerate(cycles):
            # self.pre_cycle
            [callback.pre_cycle(self) for callback in self.callback_list]
            for self.i_epoch in range(self.n_epochs):
                # self.pre_epoch
                [callback.pre_epoch(self) for callback in self.callback_list]
                try:
                    self.n_batches = len(self.train_dataloader)
                except ValueError:
                    pass
                for self.i_batch, self.batch in enumerate(self.train_dataloader):
                    # self.pre_batch
                    [callback.pre_batch(self) for
                        callback in self.callback_list]
                    self.train_step(self.batch)
                    # self.post_batch
                    [callback.post_batch(self) for
                        callback in self.callback_list]
                    if self.break_batch_loop:
                        break
                # self.post_epoch
                [callback.post_epoch(self) for callback in self.callback_list]
                self.main_dataloader = self.train_dataloader
                if self.break_epoch_loop:
                    break
            # self.post_cycle
            [callback.post_cycle(self) for callback in self.callback_list]
            if self.break_cycle_loop:
                break
        # self.post_loop
        [callback.post_loop(self) for callback in self.callback_list]
        # final cleanup
        for item in [self.i_cycle, self.n_epochs, self.i_epoch, self.i_batch]:
            del item
        del self.n_cycles
        del self.train_dataloader
        del self.valid_dataloader
        del self.callback_list
        del self.loss
        del self.main_dataloader

    @AssignAndCleanCallbacks(
        **dict(print_on_batch=1, print_epoch=False,
               print_cycle=False, ma_length=1, print_loss=True,
               print_accuracy=True, loss_format='%0.3f',
               metric_dict=None, std_to_file=None))
    def test_on_loader(
            self, test_dataloader, callbacks=None, **kwargs):
        self.model.eval()
        self.test_dataloader = test_dataloader
        self.main_dataloader = test_dataloader

        self.n_cycles = 1
        self.i_cycle = 0

        self.n_epochs = 1
        self.i_epoch = 0

        try:
            self.n_batches = len(test_dataloader)
        except ValueError:
            self.n_batches = 0

        with _torch.no_grad():
            # self.pre_loop
            [callback.pre_loop(self) for callback in self.callback_list]
            for self.i_batch, self.batch in enumerate(self.test_dataloader):
                # self.pre_batch
                [callback.pre_batch(self) for
                    callback in self.callback_list]
                self.test_step(self.batch)
                # self.post_batch
                [callback.post_batch(self) for
                    callback in self.callback_list]
            # self.post_loop
            [callback.post_loop(self) for callback in self.callback_list]

        del self.main_dataloader
        del self.callback_list

    def test_step(self, data_dict):
        self.run_extractor_hooks(data_dict)
        self.x, self.y_true = self.data_extractor(data_dict)
        # self.x = self.x.to(self.device)
        self.y_true = self.y_true.to(self.device)
        self.y_pred = self.model(self.x)
        self.loss = self.criterion(self.y_pred, self.y_true)
        self.accuracy = self.get_accuracy()

    def train_step(self, data_dict):
        self.run_extractor_hooks(data_dict)
        self.x, self.y_true = self.data_extractor(data_dict)
        self.optimizer.zero_grad()
        self.y_pred = self.model(self.x)
        self.loss = self.criterion(self.y_pred, self.y_true)
        self.loss.backward()
        self.optimizer.step()
        self.accuracy = self.get_accuracy()

    def run_extractor_hooks(self, data_dict):
        if self.pre_extractor_hooks:
            for hook in self.pre_extractor_hooks:
                hook(self, data_dict)

    def register_pre_extractor_hook(self, data_dict):
        self.pre_extractor_hooks.append(data_dict)

    def data_extractor(self, data_dict):
        if self.x_key is None:
            vals = data_dict.values()
            data_iter = iter(vals)
            x = next(data_iter)
            y = next(data_iter)
            return x.float(), y.float()
        else:
            return data_dict[self.x_key].float(), data_dict[self.y_key].float()

    def save_state(self, file_path):
        _torch.save(
            dict(
                model=self.model.state_dict(),
                optimizer=self.optimizer.state_dict()
            ),
            file_path
        )

    def load_state(self, file_path):
        checkpoint = _torch.load(file_path, map_location=self.device)
        for key, value in checkpoint.values():
            if 'model' in key.lower():
                self.model.load_state_dict(value)
            if 'optimizer' in key.lower():
                self.optimizer.load_state_dict(value)

    def load_best_model(
            self, directory, score='validation_score', model_name='model'):
        model_loader = nn.init.LoadBest(
            directory, score=score, model_name=model_name)
        self.model = model_loader(self.model)

    def get_accuracy(self):
        with _torch.no_grad():
            t = self.y_true
            p = self.y_pred
            numerator = 2 * _torch.abs(t-p)
            denominator = _torch.abs(t) + _torch.abs(p)
            error = numerator/denominator
        return (100-error*100).mean()


class InternalFieldTrainer(Trainer):

    def get_accuracy(self):
        with _torch.no_grad():
            t = self.y_true
            p = self.y_pred

            numerator = 2 * _torch.abs(t-p)
            denominator = _torch.abs(t) + _torch.abs(p) + 1e-7
            error = numerator/denominator

            return (100-error*100).mean()


class GraphTrainer(Trainer):

    def data_extractor(self, data_dict):
        if self.x_key is None:
            vals = data_dict.values()
            data_iter = iter(vals)
            x = next(data_iter)
            y = next(data_iter)
            return x, y
        else:
            return data_dict[self.x_key], data_dict[self.y_key]

    def test_step(self, data_dict):
        self.run_extractor_hooks(data_dict)
        self.x, self.y_true = self.data_extractor(data_dict)
        self.x = self.x.to(self.device)
        self.y_true = self.y_true.to(self.device)
        self.y_pred = self.model(self.x)
        self.loss = self.criterion(self.y_pred, self.y_true)

    def train_step(self, data_dict):
        self.run_extractor_hooks(data_dict)
        self.x, self.y_true = self.data_extractor(data_dict)
        self.optimizer.zero_grad()
        self.y_pred = self.model(self.x)
        self.loss = self.criterion(self.y_pred, self.y_true)
        self.loss.backward()
        self.optimizer.step()
