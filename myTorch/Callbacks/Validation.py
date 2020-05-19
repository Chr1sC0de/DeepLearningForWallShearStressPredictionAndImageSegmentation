from . import CallOn
import numpy as np
from collections import defaultdict
import torch
from .Checkpoints import Checkpoints
from collections import OrderedDict
import os
from pathlib import Path


def _accuracy_metric(true, pred):
    # relative accuracy
    numerator = 2 * np.abs(true-pred)
    denominator = np.abs(true) + np.abs(pred) + 1e-7
    return (100 - 100*np.clip(numerator/denominator, 0, 1)).mean()


class _ValidationCheckpoint(Checkpoints):

    def __init__(self, validation_obj, *args, **kwargs):
        super(_ValidationCheckpoint, self).__init__(
            *args, **kwargs
        )
        self.validation_obj = validation_obj

    def dict_constructor(self, env):
        return dict(
                model=env.model.state_dict(),
                optimizer=env.optimizer.state_dict(),
                loss_score=env.loss,
                accuracy_score=env.accuracy,
                validation_score=self.validation_obj.metric_log['accuracy'][-1]
            )

class Validation(CallOn):

    def __init__(
            self, on_batch=False, on_epoch=True, on_cycle=False,
            on_step=False, on_end=False, in_memory=False, to_file=None,
            checkpoint=False, filename='./Checkpoints/checkpoint',
            keep_best=5, **metrics):
        # initialize when to save state
        super(Validation, self).__init__(
            on_batch=on_batch, on_epoch=on_epoch, on_cycle=on_cycle,
            on_step=on_step, on_end=on_end
        )
        # initialize the metric dictionary
        self.metric_dict = dict(
            accuracy=_accuracy_metric
        )
        self.metric_dict.update(metrics)
        self.metric_log = defaultdict(list)
        self.in_memory = in_memory
        self.to_file = to_file
        if self.to_file is not None:
            self.to_file = Path(self.to_file)

        self.checkpoint = checkpoint
        if checkpoint:
            self.checkpointer = _ValidationCheckpoint(
                self, filename=filename)
            if keep_best:
                assert isinstance(keep_best, int), "keep_best must be an integer"
                self.keep_best = keep_best
                self.best_scores = OrderedDict()
                self.checkpointer.to_keep = keep_best

    def pre_loop(self, env):
        if hasattr(self, 'checkpointer'):
            self.checkpointer.pre_loop(env)
        if self.to_file is not None:
            if not self.to_file.parent.is_dir():
                self.to_file.parent.mkdir(parents=True)
        assert env.valid_dataloader is not None, \
            "to perform Validation a valid_dataloader must be specified"

    def method(self, env):
        self.calculate_metrics(env)
        self.print_log(env)

        if self.to_file:
            np.save(
                self.to_file, self.metric_log)
        if self.checkpoint:
            if not self.keep_best:
                self.checkpointer.save(env)
            else:
                self.save_best(env)

    def save_best(self, env):
        if not self.best_scores:
            if self.metric_log['accuracy'][-1] <= 100:
                if self.metric_log['accuracy'][-1] > 0:
                    self.best_scores[self.checkpointer.save(env)] = \
                        self.metric_log['accuracy'][-1]
        else:
            if len(self.best_scores) == self.keep_best:
                for key in self.best_scores.keys():
                    if self.metric_log['accuracy'][-1] > self.best_scores[key]:
                        if self.metric_log['accuracy'][-1] < 100:
                            if key.exists():
                                os.remove(key.as_posix())
                                self.best_scores.pop(key)
                                self.best_scores[self.checkpointer.save(env)] = \
                                    self.metric_log['accuracy'][-1]
                                break
                self.sort_best_scores()
            else:
                self.best_scores[self.checkpointer.save(env)] = \
                    self.metric_log['accuracy'][-1]
                self.sort_best_scores()

    def sort_best_scores(self):
        tmp = [(a, b) for a, b in self.best_scores.items()]
        tmp.sort(key=lambda x: x[1])
        self.best_scores = OrderedDict(tmp)

    def post_loop(self, env):
        super(Validation, self).post_loop(env)
        if self.on_end:
            if self.in_memory:
                env.validation_log = self.metric_log
            if self.to_file is not None:
                np.save(self.to_file, self.metric_log)

    def print_log(self, env):
        print("validation metrics: ", end="", flush=True)
        for key, value in self.metric_log.items():
            print("%s=%0.5f " % (key, value[-1]), end="", flush=True)
        print("\n", end="", flush=True)

    def calculate_metrics(self, env):
        with torch.no_grad():
            accumulator = defaultdict(list)

            for data in env.valid_dataloader:
                x, y = env.data_extractor(data)
                x = x.to(env.device)
                y = y.to(env.device)
                y_pred = env.model(x)
                y = y.to('cpu').numpy()
                y_pred = y_pred.to('cpu').numpy()
                for true, pred in zip(y, y_pred):
                    for key, metric in self.metric_dict.items():
                        accumulator[key].append(
                            np.abs(metric(true, pred))
                        )

            for key in self.metric_dict.keys():
                self.metric_log[key].append(
                    np.mean(accumulator[key]))

            self.metric_log['global_step'].append(self.global_step)
            try:
                self.metric_log['epoch'].append(
                    self.global_step/len(env.train_dataloader)
                )
            except ValueError:
                raise Exception(
                    'validation callback only valid during training')


class ReduceLROnValidPlateau(Validation):

    def __init__(self, *args, reduce_on=4, reduce_factor=0.1, **kwargs):
        super(ReduceLROnValidPlateau, self).__init__(
             *args, **kwargs
        )
        self.best_accuracy = 0
        self.attempts_since_best = 0
        self.reduce_on = reduce_on
        self.reduce_factor = reduce_factor

    def method(self, env):
        super(ReduceLROnValidPlateau, self).method(env)
        if self.best_accuracy < self.metric_log['accuracy'][-1]:
            self.best_accuracy = self.metric_log['accuracy'][-1]
            self.attempts_since_best = 0
        else:
            if self.attempts_since_best > self.reduce_on:
                env.optimizer.param_groups[0]['lr'] = \
                    env.optimizer.param_groups[0]['lr'] * self.reduce_factor
                self.attempts_since_best = 0
            else:
                self.attempts_since_best += 1
