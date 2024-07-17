import numpy as np
from typing import Tuple, Callable, Type
from Utils import *
from ActivationFunctions import Activation


class NeuralLayer:
    """
    This class represents a layer of the neural network structure, containing information such as the number of neurons determined by the user and the activation function to be used.

    Attributes:
        num_neurons (int): The number of neurons this layer will contain
        layer_input (numpy.nd_array): Activations from previous layer (or input data): (size of previous layer, number of examples).
                                      IMPORTANT: To initialize parameters and set layer_input variable you need to use set_layer_input function.
                                      The variable will be equal to None if you do not call the setter function.
        weights (numpy.nd_array): Weights matrix for the layer: numpy array of shape (size of current layer, size of previous layer)
        biases (numpy.nd_array): Bias vector for the layer, numpy array of shape (size of the current layer, 1)
        activation_function (Activation): Activation function to be used by all neurons in the layer
    """

    def __init__(self, num_neurons: int, activation_function: Type[Activation]):
        assert num_neurons > 0, "There must be at least 1 neuron in a neural layer."
        assert issubclass(activation_function, Activation), "Activation function must be an instance of the Activation class."
        self.num_neurons = num_neurons
        self.activation_function = activation_function()  # Creating an instance of given activation function
        self.layer_input = None

    def set_layer_input(self, layer_input: np.ndarray) -> None:
        if self.layer_input is None:
            self.layer_input = layer_input
            self.__initialize_parameters()  # Also this function initializes weights and biases with random numbers
        else:
            self.layer_input = layer_input

    def __initialize_parameters(self):
        assert self.layer_input is not None, "Layer input have not been defined yet. Please call the 'set_layer_input' function first."
        self.weights, self.biases = random_parameter_initialization(self.layer_input.shape[0], self.num_neurons)

        assert self.layer_input.shape[0] == self.weights.shape[1], f"The length of inputs and weights must be same."
        assert self.num_neurons == self.weights.shape[0], f"The number of neurons and the shape of their weights must match. Shape of weights that you entered is {self.weights.shape}, and you entered {self.num_neurons} neurons."
        assert self.num_neurons == self.biases.shape[0], f"The number of neurons and the shape of their biases must match. Shape of biases that you entered is {self.biases.shape}, and you entered {self.num_neurons} neurons."

    def linear_forward(self) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        This function implements the linear part of a layer's forward propagation
        :return: Result of linear calculation that the input of the activation function also called pre-activation parameter, and cache of the calculation
        """
        assert self.layer_input is not None, "Layer input have not been defined yet. Please call the 'set_layer_input' function first."
        cache = (self.layer_input, self.weights, self.biases)
        result = np.dot(self.weights, self.layer_input) + self.biases
        return result, cache

    def linear_activation_forward(self):
        """
        This function applies the linear calculation result from the linear_forward function to the activation function.
        :return: Result of non-linear calculation that the output of the activation function, and cache of all linear and non-linear calculations
        """
        linear_result, linear_cache = self.linear_forward()
        activation, activation_cache = self.activation_function.forward(linear_result)
        cache = (linear_cache, activation_cache)
        return activation, cache

    def __str__(self):
        if self.layer_input is None:
            return (f"The layer uses the {str(self.activation_function).split()[1]} activation function, and consisting of {self.num_neurons} neurons."
                    f"This is a neural network layer that does not know its inputs. That's why weights and biases matrices are not initialized.\n"
                    f"If you call 'set_inputs' function you can set inputs and initialize parameters of the layer. Then the layer be able to learn.")
        else:
            return (f"This is a neural network layer with {self.weights.size + self.biases.size} parameters\n"
                    f"that uses the {str(self.activation_function).split()[1]} activation function, and consisting of {self.num_neurons} neurons.\n"
                    f"Shape of weights array: {self.weights.shape}\n"
                    f"Shape of biases array: {self.biases.shape}\n"
                    f"Shape of inputs array: {self.layer_input.shape}")
