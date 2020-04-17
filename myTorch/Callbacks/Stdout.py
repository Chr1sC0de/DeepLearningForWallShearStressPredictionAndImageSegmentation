from . import Callback
import numpy as np
from pathlib import Path
import logging


logging.basicConfig(format="%(message)s")
logger = logging.getLogger()
logger.setLevel(logging.INFO)


class Stdout(Callback):
    default_metric_format = "%0.3f"
    base_format_string = 'cycle %d/%d epoch %d/%d batch %d/%d: '
    _print_kwargs = dict(
        # end="", flush=True
    )

    def __init__(
            self, print_on_batch=50, print_epoch=True, print_cycle=True,
            ma_length=3, print_loss=True, print_accuracy=True,
            loss_format='%0.3f', metric_dict=None, std_to_file=None):
        self.print_on_batch = print_on_batch
        self.print_epoch = print_epoch
        self.print_cycle = print_cycle
        self.print_loss = print_loss
        self.print_accuracy = print_accuracy
        self.metric_dict = metric_dict
        self.loss_format = loss_format
        self.ma_length = ma_length
        self.ma_loss = []
        self.loss_log = []
        self.ma_accuracy = []
        self.accuracy_log = []
        self.epoch_log = []
        self.global_step_log = []
        self.global_step = 0
        self.std_to_file = std_to_file
        if self.std_to_file is not None:
            self.std_to_file = Path(self.std_to_file)
            if not self.std_to_file.parent.is_dir():
                self.std_to_file.parent.mkdir(parents=True)

    def find_total_batches(self, env):
        n_samples = \
            len(env.main_dataloader.dataset)
        batch_size = env.main_dataloader.batch_size
        self.n_batches = n_samples//batch_size

    def pre_loop(self, env):
        try:
            self.find_total_batches(env)
        except ValueError:
            self.n_batches = 0
        self.env_hook = env
        env.stdout_hook = self

    def post_batch(self, env):
        if self.print_on_batch is not None:
            if env.i_batch % self.print_on_batch == 0:
                self.print()
        self.global_step += 1

    def post_epoch(self, env):
        if self.print_epoch:
            self.print()

    def post_cycle(self, env):
        if self.print_cycle:
            self.print()

    def post_loop(self, env):
        del env.stdout_hook
        self.print()

    def set_and_print_log(
            self, print_bool, param, log_name, ma_name, out_name):

        if param is not None:
            if print_bool:
                getattr(self, log_name).append(param)
                temp_log = getattr(self, log_name)
                if len(temp_log) > self.ma_length:
                    value = \
                        sum(temp_log[-self.ma_length:]) \
                        / self.ma_length
                    getattr(self, ma_name).append(value)
                else:
                    value = sum(temp_log)/len(temp_log)
                    getattr(self, ma_name).append(value)

                return '='.join([' %s' % out_name, self.loss_format % value])
                # logger.info(
                #         '='.join([' %s' % out_name, self.loss_format % value]),
                #         **self._print_kwargs)

    def print(self):
        cycle_epoch_batch = (
            self.env_hook.i_cycle + 1, self.env_hook.n_cycles,
            self.env_hook.i_epoch + 1, self.env_hook.n_epochs,
            self.env_hook.i_batch + 1, self.env_hook.n_batches
        )
        # detach the loss and y_pred y_true
        if hasattr(self.env_hook, 'loss'):
            loss = self.env_hook.loss.to('cpu').detach().numpy()
        else:
            loss = None
        y_pred = self.env_hook.y_pred.to('cpu').detach().numpy()
        y_true = self.env_hook.y_true.to('cpu').detach().numpy()
        if self.print_accuracy:
            accuracy = self.env_hook.accuracy.to('cpu').detach().numpy()
        else:
            accuracy = None
        logging_string = []
        # print the current cycle, epoch and batch number
        logging_string.append( self.base_format_string % cycle_epoch_batch ) 
        # logger.info(
        #     self.base_format_string % cycle_epoch_batch,
        #     **self._print_kwargs)
        # print the current loss and the current accuracy
        logging_string.append(self.set_and_print_log(
            self.print_loss, loss, 'loss_log', 'ma_loss', 'ma_batch_loss'))
        logging_string.append(self.set_and_print_log(
            self.print_accuracy, accuracy, 'accuracy_log', 'ma_accuracy',
            'ma_batch_accuracy'))

        # for all metrics in the metric dict, calculate its value and print
        if self.metric_dict is not None:
            assert isinstance(self.metric_dict, dict)
            for metric_name, metric in self.metric_dict.values():
                metric_value = metric(y_pred, y_true)
                if hasattr(metric, 'format'):
                    metric_format = [' %s', metric.format]
                else:
                    metric_format = [' %s', self.default_metric_format]
                metric_format = '='.join(metric_format)
                'print the metric'
                # logger.info(
                logging_string.append( metric_format % (metric_name, metric_value))
                    # **self._print_kwargs)

        logger.info(
            "".join(logging_string)
        )
        # logger.info('\n', **self._print_kwargs)
        self.global_step_log.append(self.global_step)
        try:
            self.epoch_log.append(
                self.global_step/len(self.env_hook.main_dataloader))
        except ValueError:
            pass

        if self.std_to_file is not None:
            np.save(
                self.std_to_file, {
                    'loss': self.loss_log,
                    'accuracy': self.accuracy_log,
                    'ma_accuracy': self.ma_accuracy,
                    'ma_loss': self.ma_loss,
                    'global_step': self.global_step_log,
                    'epoch': self.epoch_log
                }
            )
