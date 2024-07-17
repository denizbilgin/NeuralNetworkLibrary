import numpy as np
from ActivationFunctions import *
from Losses import *
from NeuralLayer import NeuralLayer
from Utils import random_parameter_initialization
from NeuralNetwork import NeuralNetwork

if __name__ == '__main__':
    my_layer = NeuralLayer(
        4,
        relu
    )
    my_layer.set_layer_input(np.array([1, 3, 2]))
    print(my_layer)
    print("=========================================")

    network_input = np.array([1, 3, 2])
    my_layer = NeuralLayer(
        4,
        relu
    )

    my_layer_2 = NeuralLayer(
        1,
        sigmoid
    )

    my_network = NeuralNetwork(
        network_input,
        [my_layer, my_layer_2]
    )
    my_network.network_forward()
    print(my_network.network_output)
    print(type(my_network.network_output))


