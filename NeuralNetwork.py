import numpy as np
from NeuralLayer import NeuralLayer


class NeuralNetwork:
    def __init__(self, network_input: np.ndarray, neural_layers: list[NeuralLayer]):
        self.network_input = network_input
        self.neural_layers = neural_layers
        self.network_output = None

    def network_forward(self):
        caches = []
        activation_previous = self.network_input

        for i, layer in enumerate(self.neural_layers):
            layer.set_layer_input(activation_previous)
            activation, cache = layer.linear_activation_forward()
            caches.append(cache)
            activation_previous = activation

        self.network_output = activation_previous