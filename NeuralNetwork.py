import numpy as np
from NeuralLayer import NeuralLayer
from typing import Tuple, Callable


class NeuralNetwork:
    def __init__(self, network_input: np.ndarray, neural_layers: list[NeuralLayer], cost_function: Callable):
        self.network_input = network_input
        self.neural_layers = neural_layers
        self.cost_function = cost_function

    def network_forward(self) -> Tuple[np.ndarray, list]:
        caches = []
        activation_previous = self.network_input

        for i, layer in enumerate(self.neural_layers):
            layer.set_layer_input(activation_previous)
            activation, cache = layer.linear_activation_forward()
            caches.append(cache)
            activation_previous = activation

        activation_last = activation_previous
        return activation_last, caches

    def compute_cost(self, activation_last: np.ndarray, targets: np.ndarray):
        cost = self.cost_function(targets, activation_last)
        cost = np.squeeze(cost)
        return cost


