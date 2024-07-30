import numpy as np
import h5py
import matplotlib.pyplot as plt

np.random.seed(1)

def initialize_parameters_deep(layer_dims):
    """
    This function initializes parameters of multi-layered neural network
    :param layer_dims: python array (list) containing the dimensions of each layer in our network
    :return: parameters - python dictionary containing your parameters "W1", "b1", ..., "WL", "bL"
    """
    np.random.seed(1)
    parameters = {}
    L = len(layer_dims)  # number of layers in the network

    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) / np.sqrt(layer_dims[l - 1])
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))

    return parameters

def linear_forward(A, W, b):
    """
    Implement the linear part of a layer's forward propagation
    :param A: activations from previous layer (or input data): (size of previous layer, number of examples)
    :param W: weights matrix: numpy array of shape (size of current layer, size of previous layer)
    :param b: bias vector, numpy array of shape (size of the current layer, 1)
    :return: Z - the input of the activation function, also called pre-activation parameter
    """
    Z = W.dot(A) + b
    cache = (A, W, b)

    return Z, cache

def sigmoid(Z):
    """
    Computes the sigmoid of z (scalar or numpy array)
    :param Z: numpy array of any shape
    :return: A, cache
    """
    A = 1 / (1 + np.exp(-Z))
    cache = Z
    return A, cache

def sigmoid_backward(dA, cache):
    """
    Implements the backward propagation for a single SIGMOID unit
    :param dA: post-activation gradient, of any shape
    :param cache: 'Z' where we store for computing backward propagation efficiently
    :return: dZ - Gradient of the cost with respect to Z
    """
    Z = cache

    s = 1 / (1 + np.exp(-Z))
    dZ = dA * s * (1 - s)

    return dZ

def relu(Z):
    """
    Implements the RELU function
    :param Z: Output of the linear layer, of any shape
    :return: A, cache
    """
    A = np.maximum(0, Z)
    cache = Z
    return A, cache

def relu_backward(dA, cache):
    """
    Implements the backward propagation for a single RELU unit
    :param dA: post-activation gradient, of any shape
    :param cache: 'Z' where we store for computing backward propagation efficiently
    :return: dZ - Gradient of the cost with respect to Z
    """
    Z = cache
    dZ = np.array(dA, copy=True)  # just converting dZ to a correct object

    # When z <= 0, you should set dz to 0 as well.
    dZ[Z <= 0] = 0

    return dZ

def linear_activation_forward(A_prev, W, b, activation):
    """
    Implement the forward propagation for the LINEAR->ACTIVATION layer
    :param A_prev: activations from previous layer (or input data): (size of previous layer, number of examples)
    :param W: weights matrix: numpy array of shape (size of current layer, size of previous layer)
    :param b: bias vector, numpy array of shape (size of the current layer, 1)
    :param activation: the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"
    :return: A, cache
    """
    if activation == "sigmoid":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)
    elif activation == "relu":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)

    cache = (linear_cache, activation_cache)
    return A, cache

def L_model_forward(X, parameters):
    """
    Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation"
    :param X: data, numpy array of shape (input size, number of examples)
    :param parameters: output of initialize_parameters()
    :return: AL, caches
    """
    caches = []
    A = X
    L = len(parameters) // 2  # number of layers in the neural network

    # Implement [LINEAR -> RELU]*(L-1). Add "cache" to the "caches" list
    # The for loop starts at 1 because layer 0 is the input
    for l in range(1, L):
        A_prev = A
        A, cache = linear_activation_forward(A_prev, parameters["W" + str(l)], parameters["b" + str(l)], activation="relu")
        caches.append(cache)

    # Implement LINEAR -> SIGMOID. Add "cache" to the "caches" list
    AL, cache = linear_activation_forward(A, parameters["W" + str(L)], parameters["b" + str(L)], activation="sigmoid")
    caches.append(cache)

    return AL, caches

def compute_cost(AL, Y):
    """
    Implements the cost function
    :param AL: probability vector corresponding to your label predictions, shape (1, number of examples)
    :param Y: true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of examples)
    :return: cost - cross-entropy cost
    """
    m = Y.shape[1]

    # Compute loss from aL and y.
    cost = (1./m) * (-np.dot(Y,np.log(AL).T) - np.dot(1-Y, np.log(1-AL).T))

    cost = np.squeeze(cost)  # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).
    return cost

def linear_backward(dZ, cache):
    """
    Implements the linear portion of backward propagation for a single layer (layer l)
    :param dZ: Gradient of the cost with respect to the linear output (of current layer l)
    :param cache: tuple of values (A_prev, W, b) coming from the forward propagation in the current layer
    :return: dA_prev, dW, db
    """
    A_prev, W, b = cache    # fetch data to some variables
    m = A_prev.shape[1]

    dW = (1 / m) * np.dot(dZ, A_prev.T)
    db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)

    return dA_prev, dW, db

def linear_activation_backward(dA, cache, activation):
    """
    Implements the backward propagation for the LINEAR->ACTIVATION layer
    :param dA: post-activation gradient for current layer l
    :param cache: tuple of values (linear_cache, activation_cache) we store for computing backward propagation efficiently
    :param activation: the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"
    :return: dA_prev, dW, db
    """
    linear_cache, activation_cache = cache   # fetch data to some variables

    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    return dA_prev, dW, db

def L_model_backward(AL, Y, caches):
    """
    Implements the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group
    :param AL: probability vector, output of the forward propagation (L_model_forward())
    :param Y: true "label" vector (containing 0 if non-cat, 1 if cat)
    :param caches: list of caches containing:
                every cache of linear_activation_forward() with "relu" (it's caches[l], for l in range(L-1) i.e l = 0...L-2)
                the cache of linear_activation_forward() with "sigmoid" (it's caches[L-1])
    :return: grads - A dictionary with the gradients
    """
    grads = {}
    L = len(caches)  # the number of layers
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)  # after this line, Y is the same shape as AL

    # Initializing the backpropagation
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

    # Lth layer (SIGMOID -> LINEAR) gradients. Inputs: "dAL, current_cache". Outputs: "grads["dAL-1"], grads["dWL"], grads["dbL"]
    current_cache = caches[L-1]
    grads["dA" + str(L - 1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL,
                                                                                                      current_cache,
                                                                                                      activation="sigmoid")

    # Loop from l=L-2 to l=0
    for l in reversed(range(L - 1)):
        # lth layer: (RELU -> LINEAR) gradients
        # Inputs: "grads["dA" + str(l + 1)], current_cache". Outputs: "grads["dA" + str(l)] , grads["dW" + str(l + 1)] , grads["db" + str(l + 1)]

        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 1)], current_cache, activation="relu")
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads

def update_parameters(parameters, grads, learning_rate):
    """
    Updates parameters using gradient descent
    :param parameters: python dictionary containing your parameters
    :param grads: python dictionary containing your gradients, output of L_model_backward
    :param learning_rate: the learning rate of the system
    :return: parameters - python dictionary containing your updated parameters
    """
    L = len(parameters) // 2  # number of layers in the neural network

    # Update rule for each parameter. Use a for loop
    for l in range(L):
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * grads["dW" + str(l+1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * grads["db" + str(l+1)]

    return parameters

if __name__ == '__main__':
    # You'll see just tests about the functions here


    # Initializing parameters tests
    parameters = initialize_parameters_deep([5, 4, 3])
    print("W1 = " + str(parameters["W1"]))
    print("b1 = " + str(parameters["b1"]))
    print("W2 = " + str(parameters["W2"]))
    print("b2 = " + str(parameters["b2"]))
    print("--------------------------------")


