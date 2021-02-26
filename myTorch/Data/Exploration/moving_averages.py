try:
    from myTorch.Utils import load_numpy_item
except ImportError:
    from ...Utils import load_numpy_item
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib.ticker import MultipleLocator
rc('font', **{'family': 'serif', 'serif': ['Palatino']})
rc('text', usetex=True)


def _path_or_string(obj):
    if isinstance(obj, Path):
        return True
    if isinstance(obj, str):
        return True
    return False


_x_axis_keys = ['epoch', 'global_step']


class MovingAverage:
    title_defaults = dict(
        fontsize=10
    )

    def __init__(self, data, ma, x_data=None):
        self.ma = ma
        self.raw_data = data
        self.x_data = x_data if x_data is not None else list(range(len(data)))
        self.ma_data = []

        for i in range(len(data)):
            if i < ma:
                self.ma_data.append(
                    sum(data[0:(i+1)])/(i+1)
                )
            else:
                self.ma_data.append(
                    sum(data[(i-ma):i])/ma
                )

    def plot_raw(self, *args, **kwargs):
        plt.plot(self.x_data, self.raw_data, *args, **kwargs)
        self.ax = plt.gca()
        self.f = plt.gcf()
        return self

    def plot_ma(self, *args, **kwargs):
        plt.plot(self.x_data, self.ma_data, *args, **kwargs)
        self.ax = plt.gca()
        return self

    def log_yscale(self, **kwargs):
        plt.yscale('log', **kwargs)
        return self

    def set_title(self, title, *args, **kwargs):
        self.title_defaults.update(kwargs)
        if title is not None:
            self.ax.set_title(title, *args, **self.title_defaults)
        return self

    def minor_ticks(self, n_ticks=5):
        # self.ax.xaxis.set_minor_locator(MultipleLocator(n_ticks))
        # self.ax.yaxis.set_minor_locator(MultipleLocator(n_ticks))
        self.ax.minorticks_on()

    def remove_ticks(self):
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        return self


class MADict(MovingAverage):
    label_defaults = dict(
        fontsize=8
    )

    def __init__(self, dictionary, ma, *, x_key=None, y_key):
        if _path_or_string(dictionary):
            dictionary = load_numpy_item(dictionary)
        assert isinstance(dictionary, dict), \
            f'first argument must be of type {dict}'
        x_data = dictionary[x_key] if x_key is not None else None
        self.y_key = y_key
        self.x_key = x_key
        super(MADict, self).__init__(
            dictionary[y_key], ma, x_data=x_data
        )

    def plot_raw(self, *args, **kwargs):
        if self.y_key is not None:
            kwargs['label'] = kwargs.get('label', self.y_key)
        return super(MADict, self).plot_raw(
            *args, **kwargs
        )

    def plot_ma(self, *args, **kwargs):
        if self.y_key is not None:
            kwargs['label'] = \
                kwargs.get('label', '%s ma=%d' % (self.y_key, self.ma))
        return super(MADict, self).plot_ma(
            *args, **kwargs
        )

    def label_axes(self, *args, **kwargs):
        self.label_defaults.update(kwargs)
        x_key = self.x_key
        y_key = self.y_key
        if len(args) == 1:
            x_key = args[0]
        if len(args) == 2:
            x_key = args[0]
            y_key = args[1]
        self.ax.set_xlabel(x_key, **self.label_defaults)
        self.ax.set_ylabel(y_key, **self.label_defaults)
        return self

    def set_title(self, *args, **kwargs):

        if len(args) == 0:
            return super(MADict, self).set_title(
                f'{self.x_key} vs {self.y_key}',
                *args, **kwargs)
        else:
            return super(MADict, self).set_title(
                *args, **kwargs)

    def ez_plot(self):
        self.plot_ma()
        self.plot_raw(alpha=0.3)
        self.label_axes()
        self.set_title()


class MADictMulti:
    def __init__(self, data_dictionary, ma, *keys, x_key=None):
        self.keys = keys
        for key in keys:
            setattr(
                self, key, MADict(
                    data_dictionary, ma, x_key=x_key, y_key=key)
            )

    def plot_ma_all(self, *args, **kwargs):
        for key in self.keys:
            plotter = getattr(self, key)
            plotter.plot_ma(
                *args, **kwargs)

    def plot_raw_all(self, *args, **kwargs):
        for key in self.keys:
            plotter = getattr(self, key)
            plotter.plot_raw(
                *args, **kwargs)