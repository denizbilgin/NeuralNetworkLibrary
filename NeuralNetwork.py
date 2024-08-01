import time

import numpy as np
from NeuralLayer import NeuralLayer
from typing import Tuple, List, Union, Callable, Dict
from Losses import Loss


class NeuralNetwork:
    """
    Initializes the neural network with given input, layers, loss function, targets, and learning rate.
    :param network_input: Input data for the network.
                          Please provide a numpy array that has a shape of (number of examples, number of features)
    :param targets: Target values for the input data.
                    For binary classification: numpy array with shape (number of examples, 1).
                    For multiclass classification: numpy array with shape (number of examples, number of classes).
    :param neural_layers: List of neural layers in the network.
    :param loss_function: Loss function to be used for training.
    :param learning_rate: Learning rate for gradient descent updates.
    :param training_time: Total time (seconds) spent during training.
    :param network_output: Output of the last activation function of the network after forward propagation.
                           For binary classification: shape (1, number of examples).
                           For multiclass classification: shape (number of classes, number of examples).
    """
    def __init__(self, network_input: np.ndarray, targets: np.ndarray, neural_layers: List[NeuralLayer],
                 loss_function: Loss, metrics: List[Callable], learning_rate: float = 0.01):
        assert len(neural_layers) > 0, "The network must have at least one neural layer."
        assert network_input.size > 0 and targets.size > 0, "Network input and targets cannot be empty."
        assert network_input.ndim > 1, f"The shape of the network input must be at least 2-dimensional. You give a shape of {network_input.shape}"
        assert targets.ndim > 1, f"The shape of the targets must be at least 2-dimensional. You give a shape of {targets.shape}"

        self.network_input = network_input.T  # Assigning with Transpose to reach shape of (number of features, number of examples)
        self.targets = targets.T              # Assigning with Transpose to reach shape of (number of classes, number of examples)
        self.neural_layers = neural_layers
        self.loss_function = loss_function
        self.learning_rate = learning_rate
        self.network_output = None
        self.metrics = metrics
        self.__num_layers = len(self.neural_layers)
        self.training_time = 0

    def __network_forward(self) -> List[Tuple]:
        """
        Performs forward propagation through the network.
        :return: List of caches containing the linear and activation caches for each layer, used during backpropagation
        """
        caches = []
        activation_previous = self.network_input

        for i, layer in enumerate(self.neural_layers):
            # If the weights and bias are not initialized of the current layer, it initializes them; else it updates the layer inputs according to the previous activation
            layer.set_layer_input(activation_previous)
            activation, cache = layer.linear_activation_forward()
            caches.append(cache)    # Store current caches to use them while back propagation
            activation_previous = activation

        self.network_output = activation_previous
        return caches

    def __network_backward(self, caches: List[Tuple]) -> None:
        """
        Performs backward propagation through the neural network to compute gradients.
        :param caches: List of caches containing linear and activation caches for each layer, obtained from forward propagation
        """
        assert self.network_output is not None, "Output is None right now, you need to call network_forward function first."

        #targets = self.targets.reshape(self.network_output.shape)
        d_activation_last = self.loss_function.derivative(self.targets, self.network_output)    # Initializing the backpropagation

        current_cache = caches[self.__num_layers - 1]   # Getting the last layer's gradients
        d_activation_previous, d_weights, d_biases = self.neural_layers[self.__num_layers - 1].linear_activation_backward(d_activation_last, current_cache)

        # Determining derivatives of parameters of the last layer
        self.neural_layers[self.__num_layers - 1].d_weights = d_weights
        self.neural_layers[self.__num_layers - 1].d_biases = d_biases

        for i in reversed(range(self.__num_layers - 1)):
            current_cache = caches[i]
            d_activation_previous, d_weights, d_biases = self.neural_layers[i].linear_activation_backward(d_activation_previous, current_cache)
            self.neural_layers[i].d_weights = d_weights
            self.neural_layers[i].d_biases = d_biases

    def __update_parameters(self) -> None:
        """
        Updates the parameters (weights and biases) of each layer using the computed gradients and learning rate.
        """
        for layer in self.neural_layers:
            assert layer.d_weights is not None and layer.d_biases is not None, "Gradients have not been computed. Please call network_backward first."
            layer.update_parameters(self.learning_rate)

    def __compute_cost(self) -> float:
        """
        Computes the cost (loss) between predicted output and actual targets.
        :return: Cost value indicating the difference between predicted and actual values
        """
        cost = self.loss_function.forward(self.targets, self.network_output)
        return float(cost)

    def fit(self, num_epochs: int) -> list[float]:
        """
        Trains the neural network using gradient descent for a specified number of epochs.
        :param num_epochs: Number of training epochs
        """
        costs = []
        start_time = time.time()

        for epoch in range(1, num_epochs + 1):
            # Forward propagation
            caches = self.__network_forward()

            # Compute cost
            cost = self.__compute_cost()

            # Backward propagation
            self.__network_backward(caches)

            # Update parameters
            self.__update_parameters()

            costs.append(cost)

            # Calculate metric scores
            scores = {metric.__name__: metric(self.targets, self.network_output) for metric in self.metrics}
            print(f"Epoch {epoch}/{num_epochs} | Loss: {cost:.4f} |", ", ".join([f"{name}: {score:.2f}" for name, score in scores.items()]))

        self.training_time = time.time() - start_time
        hours, rem = divmod(self.training_time, 3600)
        minutes, seconds = divmod(rem, 60)

        print(f"Training completed. Final cost: {costs[-1]}")
        print(f"Training duration: {int(hours):02}:{int(minutes):02}:{seconds:02.0f}")
        return costs

    def predict(self, data_points: np.ndarray, return_probabilities: bool = False, threshold: float = 0.5) -> Union[List[float], Tuple[List[float], List[List[Tuple]]]]:
        """
        Makes a prediction based on the input data.
        :param data_points: Input data points for the prediction. Shape of the input data must be (number of examples, number of features).
        :param return_probabilities:
        :param threshold: Threshold for converting predicted probabilities to binary values in binary classification.
                          Default value is 0.5.
        :return: Array of predicted outputs of the network.
        """
        assert data_points.ndim > 1, f"The shape of the data points must be at least 2-dimensional. You give a shape of {data_points.shape}"
        assert data_points.shape[1] == self.network_input.shape[0], "The feature number of the data point to be predicted must match the feature number of the data on which the neural network model is trained."

        outputs = []
        for data_point in data_points:
            data_point = data_point.reshape(1, data_point.shape[0])
            self.network_input = data_point.T  # Assigning with Transpose to reach shape of (number of features, 1)
            self.__network_forward()
            output = float(self.network_output.flatten()[0]) if self.network_output.shape[1] == 1 else self.network_output
            outputs.append(output)

        if not return_probabilities:
            outputs = [1 if output > threshold else 0 for output in outputs]
        return outputs

    def evaluate(self, test_data: np.ndarray, test_labels: np.ndarray) -> Dict:
        """
        Evaluates the test data using the trained deep learning model.

        This method assesses the performance of the trained neural network model on the provided test data and test labels. It predicts the labels for the test data, computes various metrics, and returns a dictionary containing the evaluation results.

        :param test_data: Input data points for prediction. The shape of the input data must be (number of examples, number of features).
        :param test_labels: Target values for the input data. The shape of the input data must be (number of examples, number of classes).
        :return: A dictionary containing evaluation metric names as keys and their corresponding values as values.

        """
        assert test_data.shape[1] == self.network_input.shape[0], "The feature number of the data point to be predicted must match the feature number of the data on which the neural network model is trained."
        assert test_data.ndim > 1, f"The shape of the test data must be at least 2-dimensional. You give a shape of {test_data.shape}"
        assert test_labels.ndim > 1, f"The shape of the test labels must be at least 2-dimensional. You give a shape of {test_labels.shape}"

        outputs = self.predict(test_data)
        predictions = np.array(outputs)

        # TODO: burası multiclass classificationda düzeltilecek
        predictions = predictions.reshape(predictions.shape[0], 1).T  # Assigning with Transpose to reach shape of (number of classes, number of examples)
        test_labels = test_labels.T                                   # Assigning with Transpose to reach shape of (number of features, number of examples)

        scores = {metric.__name__: round(float(metric(test_labels, predictions)), 2) for metric in self.metrics}

        print("===============================")
        print(len(outputs), "data points evaluated.")
        for name, score in scores.items():
            print(f"\t{name}: {score}")
        print("===============================")
        return scores


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
