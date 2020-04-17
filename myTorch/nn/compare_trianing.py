import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    file1 = r'i:\CNNForCFD\KFold_LinkNet_U\fold_1\valid\log.npy'
    file2 = r'I:\CNNForCFD\KFold_FPN_U_AREA\fold_1\valid\log.npy'
    
    item1 = np.array(np.load(file1, allow_pickle=True).item()['accuracy'])
    item2 = np.array(np.load(file2, allow_pickle=True).item()['accuracy'])
    
    plt.plot(item1)
    plt.plot(item2)
    plt.show()