import numpy as np
from typing import Union

def relu(num: float) -> float:
    """
    The Rectified Linear Unit (ReLU) function.

    The function that commonly used in neural networks. Calculation
    of the ReLU is faster than others.
    :param num: A float number, typically a neuron's input value.
    :return: The ReLU of the input number.
    """
    return max(0.0, num)


def tanh(num: float) -> float:
    """
    The hyperbolic tangent (tanh) function.

    The tanh function is an activation function used in neural networks.
    It outputs values between -1 and 1, providing a smooth gradient
    and helping to center the data.
    :param num: A float number, typically a neuron's input value.
    :return: The tanh of the input number.
    """
    return (np.e**num - np.e**(-num)) / (np.e**num + np.e**(-num))


def leaky_relu(num: float, alpha: float = 0.01) -> float:
    """
    The Leaky Rectified Linear Unit (Leaky ReLU) function.

    Unlike the standard ReLU, Leaky ReLU allows a small, non-zero gradient when the
    unit is not active, which helps to mitigate the "dying ReLU" problem
    where neurons become inactive and stop learning entirely.
    :param num: A float number, typically a neuron's input value.
    :param alpha: A float number representing the slope of the function
                  for inputs less than zero. Default value is 0.01.
    :return: The Leaky ReLU of the input number.
    """
    return max(num * alpha, num)


def elu(num: float, alpha: float = 1.0) -> float:
    """
    The Exponential Linear Unit (ELU) function.

    Unlike ReLU, it has a smooth curve for negative inputs, which helps in reducing
    the vanishing gradient problem and making the network more robust.
    The ELU function outputs the input directly if it is positive,
    and alpha times (exp(num) - 1) if it is negative.
    :param num: A float number, typically a neuron's input value.
    :param alpha: A float number representing the scaling factor for
                  negative inputs. Default value is 1.0.
    :return: The ELU of the input number.
    """
    return max(alpha*(np.exp(num)-1), num)


def sigmoid(num: Union[float, np.array]) -> float:
    """
    The sigmoid function.

    The function outputs values between 0 and 1, making it useful for models that need
    to predict probabilities. The sigmoid function provides a smooth gradient,
    which helps in gradient-based optimization methods.
    :param num: A float number or numpy array, typically a neuron's input value.
    :return: The sigmoid of the input number.
    """
    return 1 / (1 + np.exp(-num))


def softmax(nums: Union[list[float], np.matrix]) -> list[float]:
    """
    The softmax function.

    The softmax function is commonly used in machine learning, especially
    in the final layer of a neural network for classification tasks. It
    converts a list of numbers (logits) into probabilities, where each
    value is between 0 and 1, and the sum of all probabilities is 1. This
    makes it suitable for tasks where a probability distribution over classes
    is desired.
    :param nums: A list of float numbers, typically representing the logits
                 from a neural network's output layer.
    :return: A list of float numbers representing the probabilities of classes.
    """
    assert len(nums) > 1, "Length of the input (nums) of the softmax function must be greater than 1."
    return np.exp(nums) / sum(np.exp(nums))
