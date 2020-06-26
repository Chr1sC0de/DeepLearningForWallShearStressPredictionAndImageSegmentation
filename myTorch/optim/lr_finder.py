import torch as _torch
from tqdm.autonotebook import tqdm
import matplotlib.pyplot as plt
from . import (
    ExponentialLR_Finder, LinearLR_Finder
)
from ..Utils import (
    ToDevice, StateCacher)
import numpy as np
import scipy.signal as sg
from scipy.interpolate import interp1d
from ..Utils import negative_to_positive_finder


class LRFinder(object):
    '''
    an implementation of the learning rate finder:
    https://arxiv.org/abs/1506.01186
    fastai/lr_find: https://github.com/fastai/fastai
    from  https://github.com/davidtvs/pytorch-lr-finder

    '''
    def __init__(self, model, optimizer, criterion,
                 device=None, memory_cache=True, cache_dir=None):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.history = dict(lr=[], loss=[])
        self.best_loss = None
        self.memory_cache = memory_cache
        self.cache_dir = cache_dir

        self.model_device = next(self.model.parameters()).device
        self.state_cacher = StateCacher(memory_cache, cache_dir=cache_dir)
        self.state_cacher.store('model', self.model.state_dict())
        self.state_cacher.store('optimizer', self.optimizer.state_dict())

        if device:
            self.device = device
        else:
            self.device = self.model_device

        self.device_setter = ToDevice(self.device)

    def reset(self):
        self.model.load_state_dict(self.state_cacher.retrieve('model'))
        self.optimizer.load_state_dict(self.state_cacher.retrieve('optimizer'))
        self.model.to(self.model_device)

    def range_test(
        self, train_loader, val_loader=None, end_lr=10, num_iter=100,
        step_mode="exp", smooth_f=0.05, diverge_th=5, disable_tqdm=True
    ):
        self.train_loader = train_loader
        self.history = dict(lr=[], loss=[])
        self.best_loss = None

        self.model.to(self.device)

        if isinstance(step_mode, str):
            if step_mode.lower() == "exp":
                self.lr_schedule_id = "exp"
                lr_schedule = ExponentialLR_Finder(
                    self.optimizer, end_lr, num_iter)
            elif step_mode.lower() == "linear":
                lr_schedule = LinearLR_Finder(self.optimizer, end_lr, num_iter)
                self.lr_schedule_id = "linear"
            else:
                raise ValueError(
                    f"expected one of (exp, linear) not {step_mode}")
        else:
            assert callable(step_mode), TypeError(
                f'{step_mode} is neither a function or a string')

        self.iterator = iter(self.train_loader)

        for iteration in tqdm(range(num_iter), disable=disable_tqdm):
            nextVals = [val.float() for val in self._next_iter()]
            loss = self._train_batch(*nextVals)

            if val_loader:
                loss = self._validate(val_loader)
            lr_schedule.step()
            self.history["lr"].append(lr_schedule.get_lr()[0])

            if iteration == 0:
                self.best_loss = loss
            else:
                if smooth_f > 0:
                    loss = smooth_f * loss + (1-smooth_f) * self.history['loss'][-1]
                if loss > self.best_loss:
                    self.best_loss = loss

            self.history["loss"].append(loss)

            if loss > diverge_th * self.best_loss:
                print("Stopping early, the loss has diverged")
                break

        if not disable_tqdm:
            print(
                f"Learning rate search finished, see the graph with {self}.plot()")
        del self.train_loader
        del self.iterator
        self.reset()

    def _train_batch(self, x, y_true):
        self.model.train()
        x, y_true = self.device_setter(x, y_true)
        self.optimizer.zero_grad()
        y_pred = self.model(x)
        loss = self.criterion(y_pred, y_true)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def _validate(self, dataloader):
        running_loss = 0
        self.model.eval()
        with _torch.no_grad():
            for data in dataloader:
                x, y_true = self.device_setter(
                    *self.data_extractor(data))
                y_pred = self.model(x)
                loss = self.criterion(y_pred, y_true)
                running_loss += loss.item()*x.size(0)

        return running_loss / len(dataloader.dataset)

    def _next_iter(self):
        try:
            x, y_true = self.data_extractor(next(self.iterator))
        except StopIteration:
            self.iterator = iter(self.train_loader)
            x, y_true = self.data_extractor(next(self.iterator))
        return x, y_true

    def data_extractor(self, data):
        iterable_data = iter(data.values())
        x = next(iterable_data)
        y = next(iterable_data)
        return x, y

    def recommend_lr(
            self, skip_start=10, skip_end=5, window_ratio=8, to_interp=100000):
        # truncate the losses and the learning rates
        lrs = np.array(self.history["lr"])
        losses = np.array(self.history["loss"])
        lrs = lrs[losses<np.inf]
        losses = losses[losses<np.inf]
        lrs = lrs[skip_start:] if skip_end == 0 else \
            lrs[skip_start: -skip_end]
        losses = losses[skip_start:] if skip_end == 0 else \
            losses[skip_start: -skip_end]
        # smooth the losses
        n_losses = len(losses)
        w_size = n_losses//window_ratio + (n_losses//window_ratio + 1) % 2
        smooth_losses = sg.savgol_filter(losses, w_size, 3)
        # interpolate the losses geometrically spaced
        losses_of_lrs = interp1d(lrs, smooth_losses, kind='linear')
        self.new_lrs = np.geomspace(min(lrs)*1.1, max(lrs) - 1, num=to_interp)
        self.new_losses = losses_of_lrs(self.new_lrs)
        # locate regions of maximum acceleration
        max_accel_loss_ids = negative_to_positive_finder(
            np.gradient(np.gradient(self.new_losses))
        )
        # estimate the location where the loss is decreasing most rapidly
        self.min_lr_id = max_accel_loss_ids[0] + \
            int(np.mean(max_accel_loss_ids) - max_accel_loss_ids[0])//2
        min_lr = self.new_lrs[self.min_lr_id]
        # find lr when loss is min
        self.min_loss_id = np.argmin(self.new_losses)
        max_lr = self.new_lrs[self.min_loss_id]
        return min_lr, max_lr

    def plot(
            self, skip_start=10, skip_end=5,
            window_ratio=8, log_lr=True, show=True):
        if skip_start < 0:
            raise ValueError(
                "skip_start cannot be negative")
        if skip_end < 0:
            raise ValueError(
                "skip_end cannot be negative")

        lrs = self.history["lr"]
        losses = self.history["loss"]
        if skip_end == 0:
            lrs = lrs[skip_start:]
            losses = losses[skip_start:]
        else:
            lrs = lrs[skip_start:-skip_end]
            losses = losses[skip_start:-skip_end]

        min_lr, max_lr = self.recommend_lr()

        plt.plot(lrs, losses)
        if log_lr:
            plt.xscale("log")

        plt.plot(
            self.new_lrs, self.new_losses, label="smooth loss"
        )
        plt.plot(
            min_lr, self.new_losses[self.min_lr_id], 'o',
            label="recommended lr"
        )
        plt.plot(
            max_lr, self.new_losses[self.min_loss_id], 'o',
            label="maximum lr"
        )
        plt.legend()
        plt.xlabel("Learning Rate")
        plt.ylabel("Loss")
        plt.title("The recommended learning rate is %0.3E" % min_lr)
        plt.show()
        print("learning rate found !!!")


class LRFinderCFD(LRFinder):
    def _train_batch(self, x, y_true):
        self.model.train()
        x, y_true = self.device_setter(x, y_true)
        self.optimizer.zero_grad()
        y_pred = self.model(x)
        loss = self.criterion(y_pred, y_true, x)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def _validate(self, dataloader):
        running_loss = 0
        self.model.eval()
        with _torch.no_grad():
            for data in dataloader:
                x, y_true = self.device_setter(
                    *self.data_extractor(data))
                y_pred = self.model(x)
                loss = self.criterion(y_pred, y_true, x)
                running_loss += loss.item()*x.size(0)

        return running_loss / len(dataloader.dataset)