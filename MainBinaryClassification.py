import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ActivationFunctions import *
from Losses import *
from Metrics import *
from NeuralLayer import NeuralLayer
from NeuralNetwork import NeuralNetwork
from PIL import Image
import h5py


def load_dataset():
    train_dataset = h5py.File('datasets/CatVNonCat/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])  # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])  # your train set labels

    test_dataset = h5py.File('datasets/CatVNonCat/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])  # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])  # your test set labels

    classes = np.array(test_dataset["list_classes"][:])  # the list of classes

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


if __name__ == '__main__':
    train_x_orig, train_y, test_x_orig, test_y, classes = load_dataset()
    train_y = train_y.T
    test_y = test_y.T

    # Testing the 25th data of the dataset
    index = 25
    plt.imshow(train_x_orig[index])
    plt.title(f"{index}th image of train_set")
    print("y = " + str(train_y[index, 0]) + ". It's a " + classes[train_y[index, 0]].decode("utf-8") + " picture.")
    plt.show()
    print("--------------------------------")

    # Getting more info about the data's shapes and sizes
    m_train = train_x_orig.shape[0]
    num_px = train_x_orig.shape[1]
    m_test = test_x_orig.shape[0]
    print("Number of training examples: " + str(m_train))
    print("Number of testing examples: " + str(m_test))
    print("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
    print("train_x_orig shape: " + str(train_x_orig.shape))
    print("train_y shape: " + str(train_y.shape))
    print("test_x_orig shape: " + str(test_x_orig.shape))
    print("test_y shape: " + str(test_y.shape))
    print("--------------------------------")

    # Reshape the training and test examples (Flattening)
    train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0],
                                           -1)  # The "-1" makes reshape flatten the remaining dimensions
    test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1)
    # Standardize data to have feature values between 0 and 1.
    train_x = train_x_flatten / 255.
    test_x = test_x_flatten / 255.
    print("train_x's shape: " + str(train_x.shape))
    print("test_x's shape: " + str(test_x.shape))
    print("--------------------------------")

    # Determine layers
    hidden1 = NeuralLayer(20, ReLU())
    hidden2 = NeuralLayer(7, ReLU())
    hidden3 = NeuralLayer(5, ReLU())
    output_layer = NeuralLayer(1, Sigmoid())

    # Create neural network
    my_network = NeuralNetwork(
        train_x, train_y,
        [hidden1, hidden2, hidden3, output_layer],
        BinaryCrossEntropy(),
        [accuracy],
        0.001
    )

    # Train the network
    costs = my_network.fit(2000)

    # Make prediction by the trained model
    predictions = my_network.predict(test_x)

    # Evaluate the test set
    my_network.evaluate(test_x, test_y)

    # Test the ANN with our own image
    my_image = "yumak2.jpg"
    my_label_y = [1]  # the true class of your image (1 -> cat, 0 -> non-cat)

    file_name = "datasets/CatVNonCat/" + my_image
    image = np.array(Image.open(file_name).resize((num_px, num_px)))
    print("Shape of the image before processing is: " + str(image.shape))
    plt.imshow(image)
    plt.show()
    image = image / 255.
    image = image.reshape((1, num_px * num_px * 3))
    print("Shape of the image after processing is: " + str(image.shape))
    prediction = my_network.predict(image, True)[0]
    print("Probability of being a cat of the image that you uploaded:", round(prediction, 2))


# TODO: Predict fonksiyonu muhtemelen regresyon ve multiclass classification için çalışmayacak onu düzelt
# TODO: Mini-batch, stocastic gradient descent, batch
# TODO: Optimizers
# TODO: Belirli bir epoch'da eğitimi durdurup modeli kaydetmek, sonra modeli load edip tekrar eğitime devam etmek.
# TODO: Sistem multi class classification'da çalışmıyor (test et)
# TODO: Regresyon düzgün çalışmıyor, büyük ihtimalle optimizer olmadığı için
# TODO: Accuracy, F1 skoru gibi metrikleri kodla
# TODO: Görselleştirme fonksiyonları kodla (cost, accuracy..)
