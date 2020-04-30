import pathlib as pt
import click
from collections import defaultdict
import numpy as np
from numpy.random import default_rng
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font', **{'family': 'serif', 'serif': ['Times New Roman']})
rc('text', usetex=True)

def default_list_dict():
    return defaultdict(list)

@click.command()
@click.argument('path', type=click.STRING)
@click.option('--s', default='Study')
@click.option('--n', default=50, type=int)
def main(path, s, n):

    main_folder = pt.Path(path)

    # originally subsample the data
    fold_samples_dict = defaultdict(default_list_dict)

    all_fold_folders = list(main_folder.glob('fold*'))

    for i, fold_folder in enumerate(all_fold_folders):

        test_folder = fold_folder/'test'

        fold_number = i + 1

        for file in test_folder.glob('*.npz'):
            data = np.load(file)

            y_pred = np.log(np.abs(data['y_pred'].flatten()))
            y_true = np.log(data['y_true'].flatten())
            # y_pred = np.abs(data['y_pred'].flatten())
            # y_true = data['y_true'].flatten()

            rng = default_rng()

            random_numbers = rng.choice(len(y_pred), size=n, replace=False)

            fold_samples_dict['fold_%d'%fold_number]['pred'].extend(
                [y_pred[v] for v in random_numbers]
            )
            fold_samples_dict['fold_%d'%fold_number]['true'].extend(
                [y_true[v] for v in random_numbers]
            )

    for fold_name in fold_samples_dict.keys():

        fold = fold_samples_dict[fold_name]

        y_true = np.array(fold['true'])
        y_pred = np.array(fold['pred'])

        slope, intercept, r_value, p_value, std_err = \
            stats.linregress(y_true, y_pred)

        fold_samples_dict[fold_name]['slope'] = slope
        fold_samples_dict[fold_name]['intercept'] = intercept
        fold_samples_dict[fold_name]['r_value'] = r_value
        fold_samples_dict[fold_name]['p_value'] = p_value
        fold_samples_dict[fold_name]['std_err'] = std_err

    mean_dict = {}

    mean_keys = ['slope', 'intercept', 'r_value', 'p_value', 'std_err']

    for key in mean_keys:
        mean_dict[key] = np.mean(
            [fold_samples_dict[k][key] for k in fold_samples_dict.keys()]
        )

    ax = plt.gca()
    m_min, m_max = -3.5, 3.5
    x = np.linspace(m_min, m_max, 100)

    for i, key in enumerate(fold_samples_dict):

        fold = fold_samples_dict[key]
        fold_id = i+1

        m, c, r = fold['slope'], fold['intercept'], fold['r_value']
        r_2 = r**2

        true = fold['true']
        pred = fold['pred']

        ax.plot(x, m*x + c, linestyle='--', linewidth=1, alpha=1,
                label='fold %d, $%0.3fx %0.3f$, $r^2=%0.3f$' %
                (fold_id, m, c, r_2))

        ax.scatter(true, pred, 10, alpha=0.5)

    mean_m = mean_dict['slope']
    mean_c = mean_dict['intercept']
    mean_r2 = mean_dict['r_value']**2

    ax.plot(x, mean_m*x + mean_c, color='red', linestyle='-', linewidth=1,
             alpha=1, label='Expected, $%0.3fx + %0.3f$, $r^2=%0.3f$' %
             (mean_m, mean_c, mean_r2))

    ax.set_aspect('equal', 'box')
    ax.set_xlim(m_min, m_max)
    ax.set_ylim(m_min, m_max)
    ax.minorticks_on()
    ax.set_title('CFD vs CNN MWSS, Expected $r^2=%0.3f$' % (mean_r2),
                 fontsize=12)
    ax.set_ylabel('Log CNN MWSS (Pa)', fontsize=10)
    ax.set_xlabel('Log CFD MWSS (Pa)', fontsize=10)

    plt.legend(fontsize=8)
    plt.show()

if __name__ == "__main__":
    main()