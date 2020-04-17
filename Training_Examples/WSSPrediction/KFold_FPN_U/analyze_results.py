import numpy as np
import pathlib as pt

if __name__ == "__main__":

    cwd = pt.Path(__file__).parent

    fold_folders = cwd.glob("fold_*")

    for fold in fold_folders:
        test_folder = 
