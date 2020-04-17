import numpy as np
import pathlib as pt

cwd = pt.Path(__file__).parent

file = r"D:\Github\myTorch\Training_Examples\WSSPrediction\KFoldAblationStudies\Study1 - Copy\fold_5\test\log.npy"

data = np.load(file, allow_pickle=True).item()

print(data['accuracy'])

pass