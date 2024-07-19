from ActivationFunctions import *
from Losses import *
from NeuralLayer import NeuralLayer
from NeuralNetwork import NeuralNetwork


if __name__ == '__main__':
    np.random.seed(1)
    X = np.random.rand(1000, 3)
    Y = np.random.rand(1000, 1)

    my_layer = NeuralLayer(
        4,
        ReLU()
    )

    my_layer_2 = NeuralLayer(
        1,
        ReLU()
    )

    my_network = NeuralNetwork(
        X, Y,
        [my_layer, my_layer_2],
        MeanSquaredError(),
        learning_rate=0.01
    )

    costs = my_network.train(10000)

# TODO: Mini-batch, stocastic gradient descent, hepsini alan batch
# TODO: Optimizers