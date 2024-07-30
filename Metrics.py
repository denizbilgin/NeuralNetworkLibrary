import numpy as np


def accuracy(targets: np.ndarray, predictions: np.ndarray, threshold: float = 0.5) -> float:
    """
    Calculates the accuracy of the predictions against the targets.

    This function supports both binary and multiclass classification.
    For binary classification, it applies a threshold to the predictions to convert
    probabilities into binary values (0 or 1). For multiclass classification,
    it uses the argmax function to determine the predicted class.
    :param targets: Target values for the input data.
                    For binary classification, this should be a numpy array of shape (1, number of examples).
                    For multiclass classification, this should be a numpy array of shape (number of classes, number of examples).
    :param predictions: Predicted values from the network.
                        For binary classification, this should be a numpy array of shape (1, number of examples).
                        For multiclass classification, this should be a numpy array of shape (number of classes, number of examples).
    :param threshold: Threshold for converting predicted probabilities to binary values in binary classification.
                      Default value is 0.5.
    :return: The accuracy of the predictions as a percentage.
    """
    assert targets.size > 0 and predictions.size > 0, "The size of targets and predictions arrays must be bigger than 0."
    assert targets.size == predictions.size, "The size of targets and predictions arrays must be same."

    if predictions.shape[0] == 1:
        predictions = (predictions > threshold).astype(int)    # Binary classification
    else:
        predictions = np.argmax(predictions, axis=0)            # Multi class classification
        targets = np.argmax(targets, axis=0)

    return (np.sum(targets == predictions) / targets.size) * 100
