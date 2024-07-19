import numpy as np
import pandas as pd
from ActivationFunctions import *
from Losses import *
from NeuralLayer import NeuralLayer
from NeuralNetwork import NeuralNetwork


if __name__ == '__main__':
    boston_housing = pd.read_csv("datasets/BostonHousing.csv").dropna()
    Y = boston_housing["medv"].apply(lambda x: 0 if x < 23 else 1).to_numpy()
    X = boston_housing.drop(columns=["medv"]).to_numpy()
    test_data_X = X[0]
    test_data_X = test_data_X.reshape(test_data_X.shape[0], 1).T
    test_data_Y = Y[0]

    my_layer1 = NeuralLayer(
        4,
        ReLU()
    )

    my_layer_2 = NeuralLayer(
        1,
        Sigmoid()
    )

    my_network = NeuralNetwork(
        X, Y,
        [my_layer1, my_layer_2],
        BinaryCrossEntropy(),
        learning_rate=0.001
    )
    costs = my_network.train(100)
    print(my_network)
    print("Prediction of network", my_network.predict(test_data_X))
    print("Real value of data point:", test_data_Y)

# TODO: Mini-batch, stocastic gradient descent, hepsini alan batch
# TODO: Optimizers
# TODO: Belirli bir epoch'da eğitimi durdurup modeli kaydetmek, sonra modeli load edip tekrar eğitime devam etmek.
