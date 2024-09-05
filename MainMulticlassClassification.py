import os
import matplotlib.pyplot as plt
import numpy as np
from typing import List

import pandas as pd

import ActivationFunctions
from Utils import one_hot_encode
from PIL import Image
from NeuralLayer import NeuralLayer
from NeuralNetwork import NeuralNetwork
from ActivationFunctions import *
from Losses import *
from Metrics import *
from Utils import *


if __name__ == '__main__':
    dataset = pd.read_csv("datasets/StudentPerformance.csv")
    train, test = split_train_test(dataset)

    y_train = train[["GradeClass"]]
    x_train = train.drop(["GradeClass"], axis=1).to_numpy()
    y_test = test[["GradeClass"]]
    x_test = test.drop(["GradeClass"], axis=1).to_numpy()

    print("Shape of sets:")
    print("\tx_train:", x_train.shape)
    print("\ty_train:", y_train.shape)
    print("\tx_test: ", x_test.shape)
    print("\ty_test: ", y_test.shape)
    print("--------------------------------")

    # One-Hot-Encoding
    y_train, classes = one_hot_encode(y_train)
    y_test, _ = one_hot_encode(y_test)

    # Determine layers
    hidden1 = NeuralLayer(8, ReLU())
    hidden2 = NeuralLayer(4, ReLU())
    output_layer = NeuralLayer(len(classes), Softmax())

    # Determine the model
    my_network = NeuralNetwork(
        x_train, y_train,
        [hidden1, hidden2, output_layer],
        CategoricalCrossEntropy(),
        [accuracy],
        0.03
    )

    # Normalize the model
    x_train = normalize_data(x_train)
    x_test = normalize_data(x_test)
    
    my_network.fit(2500)
