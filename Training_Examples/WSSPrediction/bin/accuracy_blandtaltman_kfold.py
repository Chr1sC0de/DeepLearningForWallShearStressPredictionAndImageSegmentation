
import pathlib as pt
import click
from collections import defaultdict
import numpy as np
from numpy.random import default_rng
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib import rc
import seaborn as sns
import pingouin
from matplotlib import transforms
from scipy.stats import iqr
from numpy import median

rc('font', **{'family': 'serif', 'serif': ['Times New Roman']})
rc('text', usetex=True)

def default_list_dict():
    return defaultdict(list)

def relative_error(y_true, y_pred):
    numerator = 2 * np.abs(y_true-y_pred)
    denominator = np.abs(y_true+y_pred) + 1e-7
    return np.clip(numerator/denominator, 0, 1)

def relative_percentage_accuracy(y_true, y_pred):
    error = relative_error(y_true, y_pred)
    return (1-error)*100

def plot_blandaltman(x, y,  figsize=(5, 4),
                     dpi=100, ax=None):
    # Safety check
    x = np.asarray(x)
    y = np.asarray(y)
    assert x.ndim == 1 and y.ndim == 1
    assert x.size == y.size
    n = x.size
    mean = np.vstack((x, y)).mean(0)
    diff = x - y

    # get the interquartile range
    q975, q025 = np.percentile(diff, 97.5), np.percentile(diff, 2.5)
    q75 = np.percentile(diff, 77.5)
    q25 = np.percentile(diff, 25)
    md = median(diff)

    # Start the plot
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)

    # Plot the mean diff, limits of agreement and scatter
    ax.axhline(md, color='#6495ED', linestyle='--')
    ax.axhline(q975, color='coral', linestyle='--')
    ax.axhline(q025, color='coral', linestyle='--')
    ax.axhline(q75, color='red', linestyle='--')
    ax.axhline(q25, color='red', linestyle='--')
    ax.scatter(mean, diff, 20, alpha=0.5)

    offset = (min(q975-md, np.abs(md-q025)) / 100.0) * 1.5

    trans = transforms.blended_transform_factory(ax.transAxes, ax.transData)

    ax.text(1.03, md, '%.2f Median' % md, ha="left", va="center",
            transform=trans)

    ax.text(1.03, q975,
            '%.2f P97.5' % q975, ha="left", va="center",
            transform=trans)

    ax.text(1.03, q025,
            '%.2f P2.5' % q025, ha="left", va="center",
            transform=trans)

    ax.text(0.98, q25-offset-0.1, '%.2f P25'%q25, ha="right", va="top", transform=trans)
    ax.text(0.98, q75+offset, '%.2f P75'%q75, ha="right", va="bottom", transform=trans)

    # Labels and title
    ax.set_ylabel('Difference between methods')
    ax.set_xlabel('Mean of methods')
    ax.set_title('Bland-Altman plot')

    plt.locator_params(axis='y', nbins=20)
    plt.locator_params(axis='x', nbins=20)

    # Despine and trim
    sns.despine(trim=True, ax=ax)

    return ax

@click.command()
@click.argument('path', type=click.STRING)
@click.option('--s', default='Study')
@click.option('--n', default=50, type=int)
def main(path, s, n):

    main_folder = pt.Path(path)

    # originally subsample the data
    fold_samples_dict = defaultdict(default_list_dict)

    for i, fold_folder in enumerate(main_folder.glob('fold*')):

        test_folder = fold_folder/'test'

        fold_number = i + 1

        for file in test_folder.glob('*.npz'):
            data = np.load(file)

            y_pred = data['y_pred'].flatten()
            y_true = data['y_true'].flatten()

            rng = default_rng()

            random_numbers = rng.choice(len(y_pred), size=n, replace=False)

            fold_samples_dict['fold_%d'%fold_number]['pred'].extend(
                [y_pred[i] for i in random_numbers]
            )
            fold_samples_dict['fold_%d'%fold_number]['true'].extend(
                [y_true[i] for i in random_numbers]
            )

    accuracy_percentage_dict = defaultdict(list)

    for key in fold_samples_dict.keys():

        fold = fold_samples_dict[key]
        y_pred = np.array(fold['pred'])
        y_true = np.array(fold['true'])

        accuracy_percentage_dict['acc'].extend(
            relative_percentage_accuracy(y_true, y_pred))
        accuracy_percentage_dict['y_true'].extend(y_true)
        accuracy_percentage_dict['y_pred'].extend(y_pred)

    acc = accuracy_percentage_dict['acc']
    true = accuracy_percentage_dict['y_true']
    pred = accuracy_percentage_dict['y_pred']

    # ax = plot_blandaltman(
    #     true, pred
    # )
    ax = plot_blandaltman(
        true, pred
    )
    ax.set_title(
        'CFD vs CNN MWSS (Pa)'
    )
    ax.set_xlabel(
        '(CFD+CNN)/2'
    )
    ax.set_ylabel(
        'CFD - CNN'
    )
    ax.set_aspect('equal')
    plt.show()

if __name__ == "__main__":
    main()