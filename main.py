import numpy as np
from ActivationFunctions import *
from Losses import *
import time

if __name__ == '__main__':
    scores2D = np.array([[1, 2, 3, 6],
                         [2, 4, 5, 6],
                         [3, 8, 7, 6]])
    print(softmax(scores2D))
    print("========================")

