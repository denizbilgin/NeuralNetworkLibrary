import numpy as np
from NeuralLayer import NeuralLayer
from typing import Tuple
from Losses import Loss


class NeuralNetwork:
    def __init__(self, network_input: np.ndarray, neural_layers: list[NeuralLayer], loss_function: Loss):
        self.network_input = network_input
        self.neural_layers = neural_layers
        self.loss_function = loss_function

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

    def network_backward(self, activation_last: np.ndarray, targets: np.ndarray, caches: list[tuple]):
        gradients = {}
        targets = targets.reshape(activation_last.shape)

        # initializing the backpropagation
        # d_activation_last = - (np.divide())

    def compute_cost(self, activation_last: np.ndarray, targets: np.ndarray):
        cost = self.loss_function.forward(targets, activation_last)
        cost = np.squeeze(cost)
        return cost
