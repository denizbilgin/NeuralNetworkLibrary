import numpy as np
# TODO: Her fonksiyona açıklama yaz


def random_parameter_initialization(num_inputs: int, neuron_counts: list[int]) -> dict:
    parameters = {}
    neuron_counts.insert(0, num_inputs)
    num_layers = len(neuron_counts)

    for layer_index in range(1, num_layers):
        parameters['w' + str(layer_index)] = np.random.randn(neuron_counts[layer_index],
                                                             neuron_counts[layer_index - 1]) / np.sqrt(neuron_counts[layer_index - 1])
        parameters['b' + str(layer_index)] = np.zeros((neuron_counts[layer_index], 1))
    return parameters
