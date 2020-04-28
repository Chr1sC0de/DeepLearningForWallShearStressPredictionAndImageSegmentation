from analyze_kfold import *

@click.command()
@click.argument('path', type=click.STRING)
@click.option('--s', default='Study')
def main(path, s):

    study_folder = pt.Path(path)
    assert study_folder.exists(), 'the specified study path does not exist'
    print('analyzing', study_folder)
    save_file = study_folder/f'../{s}.csv'
    data_dict = defaultdict(list)

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

    for data_file in study_folder.glob('*.npz'):

        data = np.load(data_file)
        y_true = data['y_true'].flatten()
        y_pred = data['y_pred'].flatten()
        for key, method in name_method_dict.items():
            data_dict[key].append(
                method(y_true, y_pred)*100
            )
        for key, method in name_report_dict.items():

            limit = key.split('_')[-1]
            report = method(y_true, y_pred)
            zero_report = report['0.0']
            one_report = report['1.0']

            data_dict[f'f1-score_{limit}'].append(one_report['f1-score']*100)
            data_dict[f'precision_{limit}'].append(one_report['precision']*100)
            data_dict[f'sensitivity_{limit}'].append(one_report['recall']*100)
            data_dict[f'specificity{limit}'].append(zero_report['recall']*100)


    keys_items_cache = list(data_dict.items())

    for key, values in keys_items_cache:
        data_dict[key].append(
            (
                np.around(np.mean(values), decimals=3),
                np.around(np.std(values), decimals=3)
            )
        )

    df = pd.DataFrame(data_dict)
    df.to_csv(save_file)

if __name__ == "__main__":
    main()