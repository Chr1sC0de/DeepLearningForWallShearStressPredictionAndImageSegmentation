import numpy as np
import pathlib as pt
from scipy.stats import pearsonr
from sklearn.metrics import precision_recall_fscore_support, classification_report, r2_score
from collections import defaultdict
from functools import wraps
import pandas as pd
import click

class limit_decorator:
    def __init__(self, limit):
        self.limit=limit
    def __call__(self, function):
        @wraps(function)
        def wrapper(*args, **kwargs):
            return function(*args, **kwargs, limits=self.limit)
        return wrapper

def relative_error(y_true, y_pred):
    numerator = 2 * np.abs(y_true-y_pred)
    denominator = np.abs(y_true+y_pred) + 1e-7
    return numerator/denominator

def relative_percentage_accuracy(y_true, y_pred):
    error = relative_error(y_true, y_pred)
    return np.mean((1-error))

def get_coefficient_of_determination(y_true, y_pred):
    return r2_score(y_true, y_pred)

def get_coverage(y_true, *args ,limits=1):
    true_mask = np.zeros_like(y_true)
    true_mask[y_true < limits] = 1
    total_nodes = np.sum(np.ones_like(y_true))
    covered = np.sum(true_mask)
    return covered/total_nodes

@limit_decorator(1)
def get_coverage_1p00(*args, **kwargs):
    return get_coverage(*args, **kwargs)
@limit_decorator(0.75)
def get_coverage_0p75(*args, **kwargs):
    return get_coverage(*args, **kwargs)
@limit_decorator(0.5)
def get_coverage_0p50(*args, **kwargs):
    return get_coverage(*args, **kwargs)

def get_precision_sensitity_f1score_specificity(y_true, y_pred, limits=1):
    true_mask = np.zeros_like(y_true)
    true_mask[y_true < limits] = 1
    pred_mask = np.zeros_like(y_pred)
    pred_mask[y_pred < limits] = 1

    true_mask = true_mask.flatten()
    pred_mask = pred_mask.flatten()

    report = classification_report(true_mask, pred_mask, output_dict=True)
    return report

@limit_decorator(1)
def get_report_1p00(*args, **kwargs):
    return get_precision_sensitity_f1score_specificity(*args, **kwargs)
@limit_decorator(0.75)
def get_report_0p75(*args, **kwargs):
    return get_precision_sensitity_f1score_specificity(*args, **kwargs)
@limit_decorator(0.5)
def get_report_0p50(*args, **kwargs):
    return get_precision_sensitity_f1score_specificity(*args, **kwargs)

@click.command()
@click.argument('path', type=click.STRING)
@click.option('--s', default='Study')
def main(path, s):
    study_folder = pt.Path(path)
    assert study_folder.exists(), 'the specified study path does not exist'
    print('analyzing', study_folder)
    save_file = study_folder/f'{s}.csv'
    data_dict = defaultdict(list)

    folds = study_folder.glob('fold*')

    name_method_dict = {
        'relative_percentage_accuracy': relative_percentage_accuracy,
        'coefficient_of_determination': get_coefficient_of_determination,
        'coverage_1p00': get_coverage_1p00,
        'coverage_0p75': get_coverage_0p75,
        'coverage_0p50': get_coverage_0p50
    }
    name_report_dict = {
        'report_1p00': get_report_1p00,
        'report_0p75': get_report_0p75,
        'report_0p50': get_report_0p50,
    }

    for fold in folds:
        test_folder = fold/'test'
        data_temp_dict = defaultdict(list)
        for data_file in test_folder.glob('*.npz'):
            data = np.load(data_file)
            y_true = data['y_true'].flatten()
            y_pred = data['y_pred'].flatten()
            for key, method in name_method_dict.items():
                data_temp_dict[key].append(
                    method(y_true, y_pred)*100
                )
            for key, method in name_report_dict.items():

                limit = key.split('_')[-1]
                for key in name_method_dict.keys():
                    if limit in key:
                        coverage_key = key

                percent_coverage = data_temp_dict[key]

                if percent_coverage == 0:
                    print("no low wss")

                report = method(y_true, y_pred)
                zero_report = report['0.0']
                try:
                    one_report = report['1.0']
                except:
                    # this mean that there is no low wss along the artery
                    one_report = {'f1-score': 1.0, 'precision': 1.0, 'recall': 1.0, 'support': 0}

                data_temp_dict[f'f1-score_{limit}'].append(one_report['f1-score']*100)
                data_temp_dict[f'precision_{limit}'].append(one_report['precision']*100)
                data_temp_dict[f'sensitivity_{limit}'].append(one_report['recall']*100)
                data_temp_dict[f'specificity{limit}'].append(zero_report['recall']*100)

        for key, values in data_temp_dict.items():
            data_dict[key].append(
                (
                    np.around(np.mean(values), decimals=3),
                    np.around(np.std(values), decimals=3)
                )
            )
    keys_items_cache = list(data_dict.items())

    for key, values in keys_items_cache:
        first = [value[0] for value in values]
        second = [value[1] for value in values]
        data_dict[key].append(
            (
                np.around(np.mean(first), decimals=3),
                np.around(np.mean(second), decimals=3)
            )
        )

    df = pd.DataFrame(data_dict)
    df.to_csv(save_file)

if __name__ == "__main__":
    main()
