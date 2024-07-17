from ActivationFunctions import *
from Losses import *
from NeuralLayer import NeuralLayer
from NeuralNetwork import NeuralNetwork

if __name__ == '__main__':
    my_layer = NeuralLayer(
        4,
        ReLU()
    )
    my_layer.set_layer_input(np.array([1, 3, 2]))
    print(my_layer)
    print("=========================================")

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
        network_input,
        [my_layer, my_layer_2],
        BinaryCrossEntropy()
    )
    print(my_network.network_forward()[0])
    print("=========================================")

    targets = np.array([1, 2, 3])
    predictions = np.array([1.1, 1.9, 3.2])

    mse_loss = MeanSquaredError()
    loss_value = mse_loss.forward(targets, predictions)
    gradients = mse_loss.derivative(targets, predictions)
    print("Loss value: ", loss_value)
    print("Gradients: ", gradients)
