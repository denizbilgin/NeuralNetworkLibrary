import numpy as np
from typing import Tuple, Callable


class NeuralLayer:
    """
    This class represents a layer of the neural network structure, containing information such as the number of neurons determined by the user and the activation function to be used.

    Attributes:
        num_neurons (int): The number of neurons this layer will contain
        inputs (numpy.nd_array): Activations from previous layer (or input data): (size of previous layer, number of examples)
        weights (numpy.nd_array): Weights matrix for the layer: numpy array of shape (size of current layer, size of previous layer)
        biases (numpy.nd_array): Bias vector for the layer, numpy array of shape (size of the current layer, 1)
        activation_function (Callable): Activation function to be used by all neurons in the layer
    """

    def __init__(self, num_neurons: int, inputs: np.ndarray, weights: np.ndarray, biases: np.ndarray, activation_function: Callable):
        assert num_neurons > 0, "There must be at least 1 neuron in a neural layer."
        assert inputs.shape[0] == weights.shape[1], f"The length of inputs and weights must be same."
        assert num_neurons > 1 and num_neurons == weights.shape[0], f"The number of neurons and the shape of their weights must match. Shape of weights that you entered is {weights.shape}, and you entered {num_neurons} neurons."
        assert num_neurons > 1 and num_neurons == biases.shape[0], f"The number of neurons and the shape of their biases must match. Shape of biases that you entered is {biases.shape}, and you entered {num_neurons} neurons."

        self.num_neurons = num_neurons
        self.inputs = inputs
        self.weights = weights
        self.biases = biases
        self.activation_function = activation_function

    def linear_forward(self) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        This function implements the linear part of a layer's forward propagation
        :return: Result of linear calculation that the input of the activation function also called pre-activation parameter, and cache of the calculation
        """
        cache = (self.inputs, self.weights, self.biases)
        result = self.weights.dot(self.inputs) + self.biases
        return result, cache

    def linear_activation_forward(self):
        """
        This function applies the linear calculation result from the linear_forward function to the activation function.
        :return: Result of non-linear calculation that the output of the activation function, and cache of all linear and non-linear calculations
        """
        linear_result, linear_cache = self.linear_forward()
        activation, activation_cache = self.activation_function(linear_result)
        cache = (linear_cache, activation_cache)
        return activation, cache

    def __str__(self):
        return (f"This is a neural network layer with {self.weights.size + self.biases.size} parameters\n"
                f"that uses the {str(self.activation_function).split()[1]} activation function, consisting of {self.num_neurons} neurons.\n"
                f"Shape of weights array: {self.weights.shape}\n"
                f"Shape of biases array: {self.biases.shape}\n"
                f"Shape of inputs array: {self.inputs.shape}")