from ActivationFunctions import Activation
import numpy as np
from typing import Tuple


class NeuralLayer:
    """
    This class represents a layer of the neural network structure, containing information such as the number of neurons determined by the user and the activation function to be used.

    :param num_neurons: (int) The number of neurons this layer will contain
    :param layer_input: (numpy.nd_array) Activations from previous layer (or input data): (size of previous layer, number of examples).
                                      IMPORTANT: To initialize parameters and set layer_input variable you need to use set_layer_input function.
                                      The variable will be equal to None if you do not call the setter function.
    :param weights: (numpy.nd_array) Weights matrix for the layer: numpy array of shape (size of current layer, size of previous layer)
    :param biases: (numpy.nd_array) Bias vector for the layer, numpy array of shape (size of the current layer, 1)
    :param activation_function: (Activation) Activation function to be used by all neurons in the layer
    :param use_xavier_initialization: (bool) A boolean value to determine weight initialization technique for the layer. Default value is False.
    """

    def __init__(self, num_neurons: int, activation_function: Activation, use_xavier_initialization: bool = False):
        assert num_neurons > 0, "There must be at least 1 neuron in a neural layer."
        self.num_neurons = num_neurons
        self.activation_function = activation_function
        self.use_xavier_initialization = use_xavier_initialization
        self.layer_input = None
        self.weights = None
        self.biases = None
        self.d_weights = None
        self.d_biases = None

    def set_layer_input(self, layer_input: np.ndarray) -> None:
        """
        Sets the input for the layer and initializes the parameters if they haven't been initialized yet.
        :param layer_input: Input data or activations from the previous layer.
        """
        if self.layer_input is None:
            self.layer_input = layer_input
            self.__random_parameter_initialization()
        else:
            self.layer_input = layer_input

    def __random_parameter_initialization(self) -> None:
        """
        Initializes the weights and biases for the neural network layer using random values.

        The weights are initialized using a normal distribution (Gaussian distribution) with
        a mean of 0 and a standard deviation of 1, scaled by the square root of the number
        of input features to maintain a consistent variance (He initialization). This helps
        to prevent issues related to vanishing or exploding gradients during training.

        The biases are initialized to zero.
        """
        assert self.layer_input is not None, "Layer input has not been defined yet. Please call the 'set_layer_input' function first."
        np.random.seed(1)
        if self.use_xavier_initialization:
            self.weights = np.random.randn(self.num_neurons, self.layer_input.shape[0]) * np.sqrt(2 / (self.layer_input.shape[0] + self.num_neurons))
        else:
            self.weights = np.random.randn(self.num_neurons, self.layer_input.shape[0]) / np.sqrt(self.layer_input.shape[0])
        self.biases = np.zeros((self.num_neurons, 1))

    def linear_forward(self) -> np.ndarray:
        """
        This function implements the linear part of a layer's forward propagation
        :return: Tuple containing the linear output (pre-activation parameter) and the cache of the calculation.
        """
        assert self.layer_input is not None, "Layer input has not been defined yet. Please call the 'set_layer_input' function first."
        linear_output = np.dot(self.weights, self.layer_input) + self.biases  # Broadcasting for bias
        return linear_output

    def linear_activation_forward(self) -> Tuple[np.ndarray, Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], np.ndarray]]:
        """
        This function applies the linear calculation result from the linear_forward function to the activation function.
        :return: Tuple containing the activation output and the cache of all calculations.
        """
        linear_output = self.linear_forward()
        linear_cache = (self.layer_input, self.weights, self.biases)
        activation = self.activation_function.forward(linear_output)
        cache = (linear_cache, linear_output)
        return activation, cache

    def linear_backward(self, d_linear_output: np.ndarray, cache: Tuple[np.ndarray, np.ndarray, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        This function implements the backward propagation for the linear part.
        :param d_linear_output: Gradient of the cost with respect to the linear output of the current layer
        :param cache: Tuple containing values (layer_input, weights, biases) coming from the forward propagation in the current layer
        :return: Tuple containing gradients with respect to the input, weights, and biases.
        """
        previous_activation, weights, biases = cache
        num_examples = previous_activation.shape[1]

        self.d_weights = (1 / num_examples) * np.dot(d_linear_output, previous_activation.T)
        self.d_biases = (1 / num_examples) * np.sum(d_linear_output, axis=1, keepdims=True)
        d_previous_activation = np.dot(weights.T, d_linear_output)

        return d_previous_activation

    def linear_activation_backward(self, d_activation: np.ndarray, cache: Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], np.ndarray]) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], np.ndarray, np.ndarray]:
        """
        This function implements the backward propagation with the activation function.
        :param d_activation: Gradient of the loss with respect to the activation output.
        :param cache: Cached values from the forward propagation.
        :return: Tuple containing gradients with respect to the input, weights, and biases.
        """
        linear_cache, activation_cache = cache
        d_linear_output = self.activation_function.backward(d_activation, activation_cache)
        d_previous_activation = self.linear_backward(d_linear_output, linear_cache)
        return d_previous_activation, self.d_weights, self.d_biases

    def update_parameters(self, learning_rate: float) -> None:
        """
        This function updates the weights and biases of the layer using the calculated gradients and learning rate.
        :param learning_rate: Learning rate to update parameters
        """
        self.weights -= learning_rate * self.d_weights
        self.biases -= learning_rate * self.d_biases

    def __str__(self):
        """
        Returns a string representation of the NeuralLayer instance.
        """
        if self.layer_input is None:
            return (f"The layer uses the {self.activation_function.__class__.__name__} activation function and consists of {self.num_neurons} neurons. "
                    f"This is a neural network layer that does not know its inputs. Therefore, weights and biases matrices are not initialized.\n"
                    f"If you call 'set_layer_input' function, you can set inputs and initialize parameters of the layer. Then the layer will be able to learn.")
        else:
            return (f"This is a neural network layer with {self.weights.size + self.biases.size} parameters\n"
                    f"that uses the {self.activation_function.__class__.__name__} activation function and consists of {self.num_neurons} neurons.\n"
                    f"Shape of weights array: {self.weights.shape}\n"
                    f"Shape of biases array: {self.biases.shape}\n"
                    f"Shape of inputs array: {self.layer_input.shape}")
