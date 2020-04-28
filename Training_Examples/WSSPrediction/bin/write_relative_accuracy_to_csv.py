from print_relative_accuracy import relative_accuracy
import numpy as np
import pathlib as pt
import click
import pandas as pd

def write_relative_accuracy(
        cwd, write_name='output', csv_file=None, write_path=None):

    cwd = pt.Path(cwd)

    fold_folders = cwd.glob("fold_*")
    if csv_file is not None:
        df = pd.read_csv(csv_file)
    else:
        df = pd.DataFrame(
            {
                'name':[cwd],
                'accurcy_mean_fold_1':[],
                'accurcy_std_fold_1':[],
                'accurcy_mean_fold_2':[],
                'accurcy_std_fold_2':[],
                'accurcy_mean_fold_3':[],
                'accurcy_std_fold_3':[],
                'accurcy_mean_fold_4':[],
                'accurcy_std_fold_4':[],
                'accurcy_mean_fold_5':[],
                'accurcy_std_fold_5':[],
            }
        )

    for fold in fold_folders:
        test_folder = fold/'test'
        npz_files = list(test_folder.glob('*.npz'))
        all_accuracy = []
        for file in npz_files:
            data = np.load(file)
            # clip the first 5 columns due to noise
            y_true = data['y_true'][:,:,5:]
            y_pred = data['y_pred'][:,:,5:]
            accuracy = relative_accuracy(y_true, y_pred)
            all_accuracy.append(accuracy)
        mean_acc = np.mean(all_accuracy)
        std_acc = np.std(all_accuracy)
        df = df.append(

        )

if __name__ == "__main__":
    main()


