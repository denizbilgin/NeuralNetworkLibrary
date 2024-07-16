import numpy as np


def mean_squared_error(targets: np.ndarray, predictions: np.ndarray) -> float:
    """
    Calculates the Mean Squared Error (MSE) between target values and predicted values.

    MSE is a measure of the average squared difference between the estimated values and the actual value.
    It is calculated as the average of the squared differences between the target values and the predicted values.
    :param targets: Array of actual target values.
    :param predictions: Array of predicted values.
    :return: The mean squared error between the targets and predictions.
    """
    assert len(targets) and len(predictions) > 0, "The length of targets and predictions arrays must be bigger than 0."
    assert len(targets) == len(predictions), "The length of targets and predictions arrays must be same."
    return np.sum(np.square(targets - predictions)) / len(targets)


def mean_absolute_error(targets: np.ndarray, predictions: np.ndarray) -> float:
    """
    Calculates the Mean Absolute Error (MAE) between target values and predicted values.

    MAE is a measure of the average absolute difference between the estimated values and the actual value.
    It is calculated as the average of the absolute differences between the target values and the predicted values.
    :param targets: Array of actual target values.
    :param predictions: Array of predicted values.
    :return: The mean absolute error between the targets and predictions.
    """
    assert len(targets) and len(predictions) > 0, "The length of targets and predictions arrays must be bigger than 0."
    assert len(targets) == len(predictions), "The length of targets and predictions arrays must be same."
    return np.sum(np.abs(targets - predictions)) / len(targets)


def huber(targets: np.ndarray, predictions: np.ndarray, delta: float = 0.1) -> float:
    """
    Calculates the Huber loss between target values and predicted values.

    The Huber loss is a combination of the mean squared error and mean absolute error. It is less sensitive
    to outliers in data than the squared error loss. The parameter delta determines the threshold
    at which the loss function transitions from quadratic to linear.
    :param targets: Array of actual target values.
    :param predictions: Array of predicted values.
    :param delta: The threshold parameter that defines the point where the loss function changes. Default is 0.1.
    :return: The calculated Huber loss between the targets and predictions.
    """
    assert len(targets) and len(predictions) > 0, "The length of targets and predictions arrays must be bigger than 0."
    assert len(targets) == len(predictions), "The length of targets and predictions arrays must be same."
    absolute_error = np.abs(targets - predictions)
    quadratic = np.minimum(absolute_error, delta)
    linear = absolute_error - quadratic
    return float(np.mean(0.5 * quadratic ** 2 + delta * linear))


def binary_cross_entropy(targets: np.ndarray, predictions: np.ndarray, epsilon: float = 1e-15) -> float:
    """
    Calculates the Binary Cross-Entropy (BCE) loss between target values and predicted probabilities.

    Binary Cross-Entropy is a loss function commonly used in binary classification problems.
    It measures the dissimilarity between the true labels and predicted probabilities, and it is sensitive
    to the predicted probabilities being exactly 0 or 1, which can lead to undefined logarithms.
    The epsilon parameter is used to clip the predictions to avoid these issues.
    :param targets: Array of actual binary target values (0 or 1).
    :param predictions: Array of predicted probabilities (between 0 and 1).
    :param epsilon: mall value to clip the predictions and avoid log(0). Default is 1e-15.
    :return: The calculated binary cross-entropy loss between the targets and predictions.
    """
    assert len(targets) and len(predictions) > 0, "The length of targets and predictions arrays must be bigger than 0."
    assert len(targets) == len(predictions), "The length of targets and predictions arrays must be same."
    assert np.all(np.isin(targets, [0, 1])), "Targets must contain only binary values (0 or 1)."
    assert np.all((predictions >= 0) and (predictions <= 1)), "Predictions must be in the range [0, 1]."
    predictions = np.clip(predictions, epsilon, 1 - epsilon)
    return -(targets * np.log(predictions) + (1 - targets) * np.log(1 - predictions)).mean()


def categorical_cross_entropy(targets: np.ndarray, predictions: np.ndarray, epsilon: float = 1e-15) -> float:
    """
    Calculates the Categorical Cross-Entropy (CCE) loss between target values and predicted probabilities.

    Categorical Cross-Entropy is a loss function used in multi-class classification problems.
    It measures the dissimilarity between the true class probabilities (one-hot encoded) and the predicted probabilities.
    The epsilon parameter is used to clip the predictions to avoid issues with log(0).
    :param targets: Array of actual target values in one-hot encoded format.
    :param predictions: Array of predicted probabilities for each class.
    :param epsilon: Small value to clip the predictions and avoid log(0). Default is 1e-15.
    :return: The calculated categorical cross-entropy loss between the targets and predictions.
    """
    assert len(targets) and len(predictions) > 0, "The length of targets and predictions arrays must be bigger than 0."
    assert len(targets) == len(predictions), "The length of targets and predictions arrays must be same."
    assert targets.ndim and predictions.ndim == 2, "Targets and predictions arrays must be 2-dimensional (one-hot encoded)."
    assert predictions.shape[1] == targets.shape[1], "The number of classes in targets and predictions must match."
    predictions = np.clip(predictions, epsilon, 1 - epsilon)
    return -np.sum(targets * np.log(predictions)) / len(targets)
