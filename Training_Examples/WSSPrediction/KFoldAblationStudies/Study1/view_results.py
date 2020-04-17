import numpy as np

file = r"D:\Github\myTorch\Training_Examples\WSSPrediction\KFoldAblationStudy\fold_5\test\log.npy"

data = np.load(file, allow_pickle=True).item()

print(data['accuracy'])

pass