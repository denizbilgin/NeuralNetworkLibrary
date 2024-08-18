import numpy as np
from typing import Tuple


class Activation:
    """
    Abstract base class for activation functions.
    """
    def forward(self, linear_output: np.ndarray) -> np.ndarray:
        """
        Applies the activation function on the input linear_output.
        :param linear_output: An n-dimensional numpy array.
        :return: A tuple containing the ReLU of the n-dimensional numpy array, and the input (nums) as cache.
        """
        raise NotImplementedError

    def backward(self, d_activation: np.ndarray, cache: np.ndarray) -> np.ndarray:
        """
        Computes the backward pass of the activation function.
        :param d_activation: Gradient of the loss with respect to the output of the relevant function.
        :param cache: The input to the relevant function, which was saved during the forward pass.
        :return: The gradient of the loss with respect to the input to the relevant function.
        """
        raise NotImplementedError


class ReLU(Activation):
    def forward(self, linear_output: np.ndarray) -> np.ndarray:
        """
            The Rectified Linear Unit (ReLU) function.

            The ReLU function is commonly used in neural networks. Calculation
            of the ReLU is faster than other activation functions.
            :param linear_output: An n-dimensional numpy array.
            :return: A tuple containing the ReLU of the n-dimensional numpy array, and the input (linear_output) as cache.
            """
        assert linear_output.size > 0, "Size of the input (linear_output) of the relu function must be greater than 0."
        return np.maximum(0, linear_output)

    def backward(self, d_activation: np.ndarray, cache: np.ndarray) -> np.ndarray:
        """
            The backward pass for the ReLU function.

            Computes the gradient of the loss with respect to the input to the ReLU function.
            :param d_activation: Gradient of the loss with respect to the output of the ReLU function.
            :param cache: The input to the ReLU function, which was saved during the forward pass.
            :return: The gradient of the loss with respect to the input to the ReLU function.
            """
        d_linear_output = np.array(d_activation, copy=True)  # Creating a copy to avoid modifying the original array
        d_linear_output[cache <= 0] = 0
        return d_linear_output


class Tanh(Activation):
    def forward(self, linear_output: np.ndarray) -> np.ndarray:
        """
            The hyperbolic tangent (tanh) function.

            The tanh function is an activation function used in neural networks.
            It outputs values between -1 and 1, providing a smooth gradient
            and helping to center the data.
            :param linear_output: An n-dimensional numpy array.
            :return: A tuple containing the tanh of the n-dimensional numpy array, and the input (linear_output) as cache.
            """
        assert linear_output.size > 0, "Size of the input (linear_output) of the tanh function must be greater than 0."
        pos_nums_exp, neg_nums_exp = np.exp(linear_output), np.exp(-linear_output)
        return (pos_nums_exp - neg_nums_exp) / (pos_nums_exp + neg_nums_exp)

    def backward(self, d_activation: np.ndarray, cache: np.ndarray) -> np.ndarray:
        """
            The backward pass for the tanh function.

            Computes the gradient of the loss with respect to the input to the tanh function.
            :param d_activation: Gradient of the loss with respect to the output of the tanh function.
            :param cache: The input to the tanh function, which was saved during the forward pass.
            :return: The gradient of the loss with respect to the input to the tanh function.
            """
        d_linear_output = d_activation * (1 - np.square(self.forward(cache)[0]))
        return d_linear_output


class LeakyReLU(Activation):
    """
    :param alpha: A float number representing the slope of the function
                  for inputs less than zero. Default value is 0.01.
    """
    def __init__(self, alpha: float = 0.01):
        self.alpha = alpha

    def forward(self, linear_output: np.ndarray) -> np.ndarray:
        """
            The Leaky Rectified Linear Unit (Leaky ReLU) function.

            Unlike the standard ReLU, Leaky ReLU allows a small, non-zero gradient when the
            unit is not active, which helps to mitigate the "dying ReLU" problem
            where neurons become inactive and stop learning entirely.
            :param linear_output: An n-dimensional numpy array.
            :return: A tuple containing the Leaky ReLU of the n-dimensional numpy array, and the input (linear_output) as cache.
            """
        assert linear_output.size > 0, "Size of the input (linear_output) of the leaky relu function must be greater than 0."
        return np.maximum(linear_output * self.alpha, linear_output)

    def backward(self, d_activation: np.ndarray, cache: np.ndarray) -> np.ndarray:
        """
        The backward pass for the Leaky ReLU function.

        Computes the gradient of the loss with respect to the input to the Leaky ReLU function.
        :param d_activation: Gradient of the loss with respect to the output of the Leaky ReLU function.
        :param cache: The input to the Leaky ReLU function, which was saved during the forward pass.
        :return: The gradient of the loss with respect to the input to the Leaky ReLU function.
        """
        d_linear_output = np.array(d_activation, copy=True)  # Creating a copy to avoid modifying the original array
        d_linear_output[cache <= 0] *= self.alpha
        return d_linear_output


class ELU(Activation):
    """
    :param alpha: A float number representing the scaling factor for
                  negative inputs. Default value is 1.0.
    """
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def forward(self, linear_output: np.ndarray) -> np.ndarray:
        """
            The Exponential Linear Unit (ELU) function.

            Unlike ReLU, it has a smooth curve for negative inputs, which helps in reducing
            the vanishing gradient problem and making the network more robust.
            :param linear_output: An n-dimensional numpy array.
            :return: A tuple containing the ELU of the n-dimensional numpy array, and the input (linear_output) as cache.
            """
        assert linear_output.size > 0, "Size of the input (linear_output) of the elu function must be greater than 0."
        return np.maximum(self.alpha * (np.exp(linear_output) - 1), linear_output)

    def backward(self, d_activation: np.ndarray, cache: np.ndarray) -> np.ndarray:
        """
            The backward pass for the ELU function.

            Computes the gradient of the loss with respect to the input to the ELU function.
            :param d_activation: Gradient of the loss with respect to the output of the Leaky ELU function.
            :param cache: The input to the Leaky ELU function, which was saved during the forward pass.
            :return: The gradient of the loss with respect to the input to the ELU function.
            """
        d_linear_output = np.array(d_activation, copy=True)  # Creating a copy to avoid modifying the original array
        d_linear_output[cache <= 0] *= self.alpha * np.exp(cache)
        return d_linear_output


class Sigmoid(Activation):
    def forward(self, linear_output: np.ndarray) -> np.ndarray:
        """
        The sigmoid function.

        The function outputs values between 0 and 1, making it useful for models that need
        to predict probabilities. The sigmoid function provides a smooth gradient,
        which helps in gradient-based optimization methods.
        :param linear_output: An n-dimensional numpy array.
        :return: The sigmoid of the n-dimensional numpy array, and cache.
        """
        assert linear_output.size > 0, "Size of the input (linear_output) of the sigmoid function must be greater than 0."
        return 1 / (1 + np.exp(-linear_output))

    def backward(self, d_activation: np.ndarray, cache: np.ndarray) -> np.ndarray:
        """
            The backward pass for the sigmoid function.

            Computes the gradient of the loss with respect to the input to the sigmoid function.
            :param d_activation: Gradient of the loss with respect to the output of the sigmoid function.
            :param cache: The input to the sigmoid function, which was saved during the forward pass.
            :return: The gradient of the loss with respect to the input to the sigmoid function.
            """
        sigmoid_result = self.forward(cache)[0]
        d_linear_output = d_activation * sigmoid_result * (1 - sigmoid_result)
        return d_linear_output


class Softmax(Activation):
    def forward(self, linear_output: np.ndarray) -> np.ndarray:
        """
        The stable softmax function.

        The softmax function is commonly used in machine learning, especially
        in the final layer of a neural network for classification tasks. It
        converts a list of numbers (logits) into probabilities, where each
        value is between 0 and 1, and the sum of all probabilities is 1. This
        makes it suitable for tasks where a probability distribution over classes
        is desired.
        :param linear_output: An n-dimensional numpy array.
        :return: A numpy n-dimensional array of numbers representing the probabilities of classes, and cache.
        """
        assert linear_output.size > 1, "Length of the input (linear_output) of the softmax function must be greater than 1."
        shifted_logits = linear_output - np.max(linear_output, axis=-1, keepdims=True)
        pos_nums_exp = np.exp(shifted_logits)
        return pos_nums_exp / np.sum(pos_nums_exp, axis=-1, keepdims=True)

    def backward(self, d_activation: np.ndarray, cache: np.ndarray) -> np.ndarray:
        """
        The backward pass for the softmax function.

        Computes the gradient of the softmax function for the backward pass.
        :param d_activation: Gradient of the loss with respect to the activation.
        :param cache: Input data that was passed to the forward softmax function.
        :return: Gradient of the loss with respect to the input of the softmax function.
        """
        softmax_result = self.forward(cache)
        d_linear_output = np.zeros_like(d_activation)

        for i in range(len(d_activation)):
            jacobian_matrix = np.diag(softmax_result[i]) - np.outer(softmax_result[i], softmax_result[i])
            d_linear_output[i] = d_activation[i].dot(jacobian_matrix)
        return d_linear_output


class Unit(Activation): 
    def forward(self, linear_output: np.ndarray) -> np.ndarray:
        """
        The Unit (Linear) activation function.

        The Unit function is used in the output layer of regression problems.
        It returns the input as it is.
        :param linear_output: An n-dimensional numpy array.
        :return: A tuple containing the ReLU of the n-dimensional numpy array, and the input (linear_output) as cache.
        """
        assert linear_output.size > 0, "Size of the input (linear_output) of the unit function must be greater than 0."
        return linear_output

    def backward(self, d_activation: np.ndarray, cache: np.ndarray) -> np.ndarray:
        """
        The backward pass for the Unit (Linear) function.

        Since the derivative of the Unit function is 1, the gradient of the loss with respect to
        the input is the same as the gradient of the loss with respect to the output.
        :param d_activation: Gradient of the loss with respect to the output of the ReLU function.
        :param cache: The input to the ReLU function, which was saved during the forward pass.
        :return: The gradient of the loss with respect to the input to the Unit function.
        """
        assert d_activation.size > 0, "Size of the input (d_activation) of the backward function must be greater than 0."
        return d_activation
