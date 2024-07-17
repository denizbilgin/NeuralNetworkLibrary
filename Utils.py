import numpy as np
from typing import Tuple
# TODO: Her fonksiyona açıklama yaz


def random_parameter_initialization(num_inputs: int, num_neurons: int) -> Tuple[np.ndarray, np.ndarray]:
    weights = np.random.randn(num_neurons, num_inputs) / np.sqrt(num_inputs)
    biases = np.zeros(num_neurons)
    return weights, biases
