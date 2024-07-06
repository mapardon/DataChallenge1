import warnings
import numpy as np

warnings.filterwarnings("error")


def sigmoid(x):
    try:
        tmp = 1 / (1 + np.exp(-x))
    except RuntimeWarning as rw:
        print(rw)
    return tmp
