import numpy as np
import pathlib as pt
import matplotlib.pyplot as plt

def relative_error(true, pred):
    numerator = 2*np.abs(true-pred)
    denominator = np.abs(true+pred+1e-7)
    return np.clip(numerator/denominator, 0, 1)

def relative_accuracy(true, pred):
    return ((1-relative_error(true,pred))*100).mean()

if __name__ == "__main__":

    cwd = pt.Path(__file__).parent

    fold_folders = cwd.glob("fold_*")

    for fold in fold_folders:
        test_folder = fold/'test'
        npz_files = list(test_folder.glob('*.npz'))

        all_accuracy = []

        for file in npz_files:
            data = np.load(file)

            # clip the first 10 columns due to noise
            y_true = data['y_true'][:,:,5:]
            y_pred = data['y_pred'][:,:,5:]

            accuracy = relative_accuracy(y_true, y_pred)

            all_accuracy.append(accuracy)

        mean_acc = np.mean(all_accuracy)
        std_acc = np.std(all_accuracy)

        print(mean_acc, std_acc)

