from ActivationFunctions import *
from Losses import *
from NeuralLayer import NeuralLayer
from NeuralNetwork import NeuralNetwork

if __name__ == '__main__':
    np.random.seed(1)
    X = np.random.rand(100, 10)  # 100 örnek, her biri 10 özellik
    Y = np.random.rand(100, 1)  # 100 örnek, her biri 1 hedef değer

    network_input = np.array([1, 3, 2])
    my_layer = NeuralLayer(
        4,
        ReLU()
    )

    my_layer_2 = NeuralLayer(
        1,
        Sigmoid()
    )

    my_network = NeuralNetwork(
        X, Y,
        [my_layer, my_layer_2],
        MeanSquaredError(),
        learning_rate=0.01
    )

    my_network.train(10)
