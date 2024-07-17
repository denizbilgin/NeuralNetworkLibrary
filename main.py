import numpy as np
from ActivationFunctions import *
from Losses import *
from NeuralLayer import NeuralLayer
from Utils import random_parameter_initialization

if __name__ == '__main__':
    scores2D = np.array([[1, 2, 3, 6],
                         [2, 4, 5, 6],
                         [3, 8, 7, 6]])
    print(softmax(scores2D))
    print("=========================================")

    # weights matrisi: (o katmandaki nöron sayısı , katmanın input sayısı)
    # inputs matrisi: (input sayısı , )
    # bias matrisi: (o katmandaki nöron sayısı , )


    print("Weights shape:")
    print(np.array([[0.7, -0.4, 0.8],
                    [0.4, -0.5, 0.1],
                    [0.1, -0.7, 0.2],
                    [0.5, -0.1, 0.1]]).shape)
    print("--------------")
    print("Inputs shape:")
    print(np.array([1, 3, 2]).shape)
    print("--------------")
    print("Biases shape:")
    print(np.array([3, 1, 2, 5]).shape)

    myLayer = NeuralLayer(
        4,
        np.array([1, 3, 2]),
        np.array([[0.7, -0.4, 0.8],
                  [0.4, -0.5, 0.1],
                  [0.1, -0.7, 0.2],
                  [0.5, -0.1, 0.1]]),
        np.array([3, 1, 2, 5]),
        relu
    )
    print(myLayer.linear_forward())
    print(myLayer)
    print("=========================================")

    params = random_parameter_initialization(3, [2])
    myLayer = NeuralLayer(
        2,
        np.array([1, 3, 2]),
        params["w1"],
        params["b1"],
        relu
    )
    print(myLayer.linear_forward())
    print(myLayer)
    print("=========================================")

    nums = np.array([[-1.0, 0.0, 1.0], [2.0, -2.0, 3.0]])
    d_activation = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    relu_output, cache = relu(nums)
    print("ReLU aktivasyonu:", relu_output)

    # Geri yayılım
    d_nums = relu_backward(d_activation, cache)
    print("ReLU geri yayılımı:", d_nums)

