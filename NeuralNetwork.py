import time

import numpy as np
from NeuralLayer import NeuralLayer
from typing import Tuple, List, Union
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
    :param training_time: Total time (seconds) spent during training.
    :param network_output: It is the output of the last activation function of the network after forward propagation is finished.
                           This variable also holds the model's predictions about all training examples after the individual model training has finished.
                           Shape of network output is (1, number of examples)
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
        self.training_time = 0

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
        start_time = time.time()
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

        self.training_time = time.time() - start_time
        hours, rem = divmod(self.training_time, 3600)
        minutes, seconds = divmod(rem, 60)

        progress_bar.close()
        print(f"Training completed. Final cost: {costs[-1]}")
        print(f"Training time: {int(hours):02}:{int(minutes):02}:{seconds:02.0f}")
        return costs

    def predict(self, inputs: np.ndarray, return_caches: bool = False) -> Union[np.ndarray, float, Tuple[np.ndarray, List[Tuple]], Tuple[float, List[Tuple]]]:
        """
        Makes a prediction based on the input data.
        :param inputs: Input data for the prediction. Shape of the input data must be (1, number of features).
        :param return_caches: If you want to reach caches of forward propagations assign it as True.
                              Default value is False.
        :return: Predicted output of the network.
        """
        assert inputs.shape[0] == 1, "Only one data point can be sent to the neural network to predict at a time."
        assert inputs.shape[1] == self.network_input.shape[0], "The feature number of the data point to be predicted must match the feature number of the data on which the neural network model is trained."
        self.network_input = inputs.T  # Assigning with Transpose to reach shape of (number of features, number of examples)
        caches = self.network_forward()
        output = float(self.network_output.flatten()[0]) if self.network_output.shape[1] == 1 else self.network_output

        if return_caches:
            return output, caches
        return output

    def __str__(self):
        """
        Returns a string representation of the NeuralNetwork object, describing its structure and parameters.
        :return: String representation of the NeuralNetwork object
        """
        description = f"=================================================\nNeural Network with {self.__num_layers} layers:\n"
        total_params = 0
        total_bytes = 0
        for i, layer in enumerate(self.neural_layers):
            layer_params_count = layer.weights.size + layer.biases.size
            description += f"\tLayer {i + 1}: {layer.num_neurons} neurons, activation {layer.activation_function.__class__.__name__}(), {layer_params_count} parameters\n"
            total_params += layer_params_count
            total_bytes += layer.weights.nbytes + layer.biases.nbytes
        total_megabytes = total_bytes / (1024 * 1024)
        description += f"Total parameters: {total_params}\nMegabytes occupied by parameters: {total_megabytes:.4f} MB\nInput shape: {self.network_input.T.shape}\nOutput shape: {self.network_output.T.shape}\n================================================="
        return description
