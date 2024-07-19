import numpy as np
from NeuralLayer import NeuralLayer
from typing import Tuple, List
from Losses import Loss
from tqdm import tqdm


class NeuralNetwork:
    """
    Initializes the neural network with given input, layers, loss function, targets, and learning rate.
    :param network_input: Input data for the network.
                          Please provide a numpy array that has a shape of (number of examples, number of features)
    :param targets: Target values for the input data.
                    Please provide a numpy array that has a shape of (number of examples, 1)
    :param neural_layers: List of neural layers in the network.
    :param loss_function: Loss function to be used for training.
    :param learning_rate: Learning rate for gradient descent updates.
    """
    def __init__(self, network_input: np.ndarray, targets: np.ndarray, neural_layers: List[NeuralLayer], loss_function: Loss, learning_rate: float = 0.01):
        assert len(neural_layers) > 0, "The network must have at least one neural layer."
        assert network_input.size > 0 and targets.size > 0, "Network input and targets cannot be empty."

        self.network_input = network_input.T  # Assigning with Transpose to reach shape of (number of features, number of examples)
        self.targets = targets.T              # Assigning with Transpose to reach shape of (1, number of examples)
        self.neural_layers = neural_layers
        self.loss_function = loss_function
        self.learning_rate = learning_rate
        self.network_output = None
        self.__num_layers = len(self.neural_layers)

    def network_forward(self) -> List[Tuple]:
        """
        Performs forward propagation through the network.
        :return: List of caches containing the linear and activation caches for each layer, used during backpropagation
        """
        caches = []
        activation_previous = self.network_input

        for i, layer in enumerate(self.neural_layers):
            layer.set_layer_input(activation_previous)
            activation, cache = layer.linear_activation_forward()
            caches.append(cache)
            activation_previous = activation

        self.network_output = activation_previous
        return caches

    def network_backward(self, caches: List[Tuple]) -> None:
        """
        Performs backward propagation through the neural network to compute gradients.
        :param caches: List of caches containing linear and activation caches for each layer, obtained from forward propagation
        """
        assert self.network_output is not None, "Output is None right now, you need to call network_forward function first."

        targets = self.targets.reshape(self.network_output.shape)

        # Initializing the backpropagation
        d_activation_last = self.loss_function.derivative(targets, self.network_output)

        # Getting the last layer's gradients
        current_cache = caches[self.__num_layers - 1]
        d_activation_previous, d_weights, d_biases = self.neural_layers[self.__num_layers - 1].linear_activation_backward(d_activation_last, current_cache)
        self.neural_layers[self.__num_layers - 1].d_weights = d_weights
        self.neural_layers[self.__num_layers - 1].d_biases = d_biases

        for i in reversed(range(self.__num_layers - 1)):
            current_cache = caches[i]
            d_activation_previous, d_weights, d_biases = self.neural_layers[i].linear_activation_backward(d_activation_previous, current_cache)
            self.neural_layers[i].d_weights = d_weights
            self.neural_layers[i].d_biases = d_biases

    def update_parameters(self) -> None:
        """
        Updates the parameters (weights and biases) of each layer using the computed gradients and learning rate.
        """
        for layer in self.neural_layers:
            layer.update_parameters(self.learning_rate)

    def compute_cost(self) -> float:
        """
        Computes the cost (loss) between predicted output and actual targets.
        :return: Cost value indicating the difference between predicted and actual values
        """
        cost = self.loss_function.forward(self.targets, self.network_output)
        return float(cost)

    def train(self, num_epochs: int) -> list[float]:
        """
        Trains the neural network using gradient descent for a specified number of epochs.
        :param num_epochs: Number of training epochs
        """
        progress_bar = tqdm(total=num_epochs, desc="Training", position=0, leave=True)
        costs = []
        for epoch in range(num_epochs):
            # Forward propagation
            caches = self.network_forward()

            # Compute cost
            cost = self.compute_cost()
            # Backward propagation
            self.network_backward(caches)

            # Update parameters
            self.update_parameters()

            costs.append(cost)
            progress_bar.set_postfix({'Cost': f'{cost:.6f}'})
            progress_bar.update(1)

        progress_bar.close()
        print(f"Training completed. Final cost: {costs[-1]}")
        return costs
    def __str__(self):
        """
        Returns a string representation of the NeuralNetwork object, describing its structure and parameters.
        :return: String representation of the NeuralNetwork object
        """
        description = f"=================================================\nNeural Network with {self.__num_layers} layers:\n"
        total_params = 0
        for i, layer in enumerate(self.neural_layers):
            layer_params_count = layer.weights.size + layer.biases.size
            description += f"\tLayer {i + 1}: {layer.num_neurons} neurons, activation {layer.activation_function.__class__.__name__}(), {layer_params_count} parameters\n"
            total_params += layer_params_count
        description += f"Total parameters: {total_params}\nInput shape: {self.network_input.shape}\n================================================="
        return description
