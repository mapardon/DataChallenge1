import numpy as np
import warnings

warnings.filterwarnings("error")


def sigmoid(x):
    try:
        return 1 / (1 + np.exp(-x))
    except RuntimeWarning as rw:  # filter potential overflows
        return 1
