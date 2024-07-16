import numpy as np
from typing import Tuple


def relu(nums: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    The Rectified Linear Unit (ReLU) function.

    The ReLU function is commonly used in neural networks. Calculation
    of the ReLU is faster than other activation functions.
    :param nums: An n-dimensional numpy array.
    :return: A tuple containing the ReLU of the n-dimensional numpy array, and the input (nums) as cache.
    """
    assert nums.size > 0, "Size of the input (nums) of the relu function must be greater than 0."
    return np.maximum(0.0, nums), nums


def relu_backward(d_activation: np.ndarray, cache: np.ndarray) -> np.ndarray:
    """
    The backward pass for the ReLU function.

    Computes the gradient of the loss with respect to the input to the ReLU function.
    :param d_activation: Gradient of the loss with respect to the output of the ReLU function.
    :param cache: The input to the ReLU function, which was saved during the forward pass.
    :return: The gradient of the loss with respect to the input to the ReLU function.
    """
    d_nums = np.array(d_activation, copy=True)  # Creating a copy to avoid modifying the original array
    d_nums[cache <= 0] = 0
    return d_nums


def tanh(nums: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    The hyperbolic tangent (tanh) function.

    The tanh function is an activation function used in neural networks.
    It outputs values between -1 and 1, providing a smooth gradient
    and helping to center the data.
    :param nums: An n-dimensional numpy array.
    :return: A tuple containing the tanh of the n-dimensional numpy array, and the input (nums) as cache.
    """
    assert nums.size > 0, "Size of the input (nums) of the tanh function must be greater than 0."
    pos_nums_exp, neg_nums_exp = np.exp(nums), np.exp(-nums)
    return (pos_nums_exp - neg_nums_exp) / (pos_nums_exp + neg_nums_exp), nums


def tanh_backward(d_activation: np.ndarray, cache: np.ndarray) -> np.ndarray:
    """
    The backward pass for the tanh function.

    Computes the gradient of the loss with respect to the input to the tanh function.
    :param d_activation: Gradient of the loss with respect to the output of the tanh function.
    :param cache: The input to the tanh function, which was saved during the forward pass.
    :return: The gradient of the loss with respect to the input to the tanh function.
    """
    d_nums = d_activation * (1 - np.square(tanh(cache)[0]))
    return d_nums


def leaky_relu(nums: np.ndarray, alpha: float = 0.01) -> Tuple[np.ndarray, np.ndarray]:
    """
    The Leaky Rectified Linear Unit (Leaky ReLU) function.

    Unlike the standard ReLU, Leaky ReLU allows a small, non-zero gradient when the
    unit is not active, which helps to mitigate the "dying ReLU" problem
    where neurons become inactive and stop learning entirely.
    :param nums: An n-dimensional numpy array.
    :param alpha: A float number representing the slope of the function
                  for inputs less than zero. Default value is 0.01.
    :return: A tuple containing the Leaky ReLU of the n-dimensional numpy array, and the input (nums) as cache.
    """
    assert nums.size > 0, "Size of the input (nums) of the leaky relu function must be greater than 0."
    return np.maximum(nums * alpha, nums), nums


def leaky_relu_backward(d_activation: np.ndarray, cache: np.ndarray, alpha: float = 0.01) -> np.ndarray:
    """
    The backward pass for the Leaky ReLU function.

    Computes the gradient of the loss with respect to the input to the Leaky ReLU function.
    :param d_activation: Gradient of the loss with respect to the output of the Leaky ReLU function.
    :param cache: The input to the Leaky ReLU function, which was saved during the forward pass.
    :param alpha: A float number representing the slope of the function
                  for inputs less than zero. Default value is 0.01.
    :return: The gradient of the loss with respect to the input to the Leaky ReLU function.
    """
    d_nums = np.array(d_activation, copy=True)  # Creating a copy to avoid modifying the original array
    d_nums[cache <= 0] *= alpha
    return d_nums


def elu(nums: np.ndarray, alpha: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    The Exponential Linear Unit (ELU) function.

    Unlike ReLU, it has a smooth curve for negative inputs, which helps in reducing
    the vanishing gradient problem and making the network more robust.
    :param nums: An n-dimensional numpy array.
    :param alpha: A float number representing the scaling factor for
                  negative inputs. Default value is 1.0.
    :return: A tuple containing the ELU of the n-dimensional numpy array, and the input (nums) as cache.
    """
    assert nums.size > 0, "Size of the input (nums) of the elu function must be greater than 0."
    return np.maximum(alpha * (np.exp(nums) - 1), nums), nums


def elu_backward(d_activation: np.ndarray, cache: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """
    The backward pass for the ELU function.

    Computes the gradient of the loss with respect to the input to the ELU function.
    :param d_activation: Gradient of the loss with respect to the output of the Leaky ELU function.
    :param cache: The input to the Leaky ELU function, which was saved during the forward pass.
    :param alpha: A float number representing the scaling factor for
                  negative inputs. Default value is 1.0.
    :return: The gradient of the loss with respect to the input to the ELU function.
    """
    d_nums = np.array(d_activation, copy=True)  # Creating a copy to avoid modifying the original array
    d_nums[cache <= 0] *= alpha * np.exp(cache)
    return d_nums


def sigmoid(nums: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    The sigmoid function.

    The function outputs values between 0 and 1, making it useful for models that need
    to predict probabilities. The sigmoid function provides a smooth gradient,
    which helps in gradient-based optimization methods.
    :param nums: An n-dimensional numpy array.
    :return: The sigmoid of the n-dimensional numpy array, and cache.
    """
    assert nums.size > 0, "Size of the input (nums) of the sigmoid function must be greater than 0."
    return 1 / (1 + np.exp(-nums)), nums


def sigmoid_backward(d_activation: np.ndarray, cache: np.ndarray) -> np.ndarray:
    """
    The backward pass for the sigmoid function.

    Computes the gradient of the loss with respect to the input to the sigmoid function.
    :param d_activation: Gradient of the loss with respect to the output of the sigmoid function.
    :param cache: The input to the sigmoid function, which was saved during the forward pass.
    :return: The gradient of the loss with respect to the input to the sigmoid function.
    """
    sigmoid_result = sigmoid(cache)[0]
    d_nums = d_activation * sigmoid_result * (1 - sigmoid_result)
    return d_nums


def softmax(nums: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    The softmax function.

    The softmax function is commonly used in machine learning, especially
    in the final layer of a neural network for classification tasks. It
    converts a list of numbers (logits) into probabilities, where each
    value is between 0 and 1, and the sum of all probabilities is 1. This
    makes it suitable for tasks where a probability distribution over classes
    is desired.
    :param nums: An n-dimensional numpy array.
    :return: A numpy n-dimensional array of numbers representing the probabilities of classes, and cache.
    """
    assert nums.size > 1, "Length of the input (nums) of the softmax function must be greater than 1."
    pos_nums_exp = np.exp(nums)
    return pos_nums_exp / sum(pos_nums_exp), nums


def softmax_backward(d_activation: np.ndarray, cache: np.ndarray) -> np.ndarray:
    """
    The backward pass for the softmax function.

    Computes the gradient of the softmax function for the backward pass.
    :param d_activation: Gradient of the loss with respect to the activation.
    :param cache: Input data that was passed to the forward softmax function.
    :return: Gradient of the loss with respect to the input of the softmax function.
    """
    softmax_result = softmax(cache)
    d_nums = np.zeros_like(d_activation)

    for i in range(len(d_activation)):
        jacobian_matrix = np.diag(softmax_result[i]) - np.outer(softmax_result[i], softmax_result[i])
        d_nums[i] = d_activation[i].dot(jacobian_matrix)
    return d_nums
