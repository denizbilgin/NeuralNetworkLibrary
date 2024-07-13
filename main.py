import numpy as np
from ActivationFunctions import *
from Losses import *

if __name__ == '__main__':
    y_true = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    y_pred = np.array([[0.8, 0.1, 0.1], [0.2, 0.7, 0.1], [0.1, 0.2, 0.7]])

    print(categorical_cross_entropy(y_true, y_pred))
