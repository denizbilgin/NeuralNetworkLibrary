import numpy as np


class Loss:
    """
    Abstract base class for loss functions.
    """
    def forward(self, targets: np.ndarray, predictions: np.ndarray) -> float:
        """
        Calculates loss on targets and predictions.
        :param targets: Array of actual target values.
        :param predictions: The mean squared error between the targets and predictions.
        :return: A float number that represents the loss.
        """
        raise NotImplementedError

    def derivative(self, targets: np.ndarray, predictions: np.ndarray) -> np.ndarray:
        """
        Calculates the gradient (derivative) of the loss function with respect to the predictions.
        :param targets: Array of actual target values.
        :param predictions: Array of predicted values.
        :return: A numpy array containing the gradients of the loss function with respect to the predictions.
                 This array will have the same shape as the input predictions.
        """
        raise NotImplementedError


class MeanSquaredError(Loss):
    """
    Mean Squared Error (MSE) loss function.
    """
    def forward(self, targets: np.ndarray, predictions: np.ndarray) -> float:
        """
        Calculates the Mean Squared Error (MSE) between target values and predicted values.

        MSE is a measure of the average squared difference between the estimated values and the actual value.
        It is calculated as the average of the squared differences between the target values and the predicted values.
        :param targets: Array of actual target values.
        :param predictions: Array of predicted values.
        :return: The mean squared error between the targets and predictions.
        """
        assert targets.size > 0 and predictions.size > 0, "The size of targets and predictions arrays must be bigger than 0."
        assert targets.size == predictions.size, "The size of targets and predictions arrays must be same."

        loss = np.sum(np.square(targets - predictions))
        return loss / predictions.shape[0]

    def derivative(self, targets: np.ndarray, predictions: np.ndarray) -> np.ndarray:
        """
        Calculates the gradient (derivative) of the Mean Squared Error (MSE) loss function
        with respect to the predictions.
        :param targets: Array of actual target values.
        :param predictions: Array of predicted values.
        :return: A numpy array containing the gradients of the MSE loss function with respect to the predictions.
                 This array will have the same shape as the input predictions.
        """
        assert targets.size > 0 and predictions.size > 0, "The size of targets and predictions arrays must be bigger than 0."
        assert targets.size == predictions.size, "The size of targets and predictions arrays must be same."

        gradient = 2 * (predictions - targets)
        return gradient / predictions.shape[0]


class MeanAbsoluteError(Loss):
    """
    Mean Absolute Error (MAE) loss function.
    """
    def forward(self, targets: np.ndarray, predictions: np.ndarray) -> float:
        """
        Calculates the Mean Absolute Error (MAE) between target values and predicted values.

        MAE is a measure of the average absolute difference between the estimated values and the actual value.
        It is calculated as the average of the absolute differences between the target values and the predicted values.
        :param targets: Array of actual target values.
        :param predictions: Array of predicted values.
        :return: The mean absolute error between the targets and predictions.
        """
        assert targets.size > 0 and predictions.size > 0, "The size of targets and predictions arrays must be bigger than 0."
        assert targets.size == predictions.size, "The size of targets and predictions arrays must be same."

        loss = np.sum(np.abs(targets - predictions))
        return loss / predictions.shape[0]

    def derivative(self, targets: np.ndarray, predictions: np.ndarray) -> np.ndarray:
        """
        Calculates the gradient (derivative) of the Mean Absolute Error (MAE) loss function
        with respect to the predictions.
        :param targets: Array of actual target values.
        :param predictions: Array of predicted values.
        :return: A numpy array containing the gradients of the MAE loss function with respect to the predictions.
                 This array will have the same shape as the input predictions.
        """
        assert targets.size > 0 and predictions.size > 0, "The size of targets and predictions arrays must be bigger than 0."
        assert targets.size == predictions.size, "The size of targets and predictions arrays must be same."

        gradient = np.sign(predictions - targets)
        return gradient / predictions.shape[0]


class Huber(Loss):
    """
    Huber's loss function.
    """
    def forward(self, targets: np.ndarray, predictions: np.ndarray, delta: float = 0.1) -> float:
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
        assert targets.size > 0 and predictions.size > 0, "The size of targets and predictions arrays must be bigger than 0."
        assert targets.size == predictions.size, "The size of targets and predictions arrays must be same."

        absolute_error = np.abs(targets - predictions)
        quadratic = np.minimum(absolute_error, delta)
        linear = absolute_error - quadratic
        loss = np.mean(0.5 * quadratic ** 2 + delta * linear)
        return float(loss)

    def derivative(self, targets: np.ndarray, predictions: np.ndarray, delta: float = 0.1) -> np.ndarray:
        """
        Calculates the gradient (derivative) of the Huber loss function with respect to the predictions.
        :param targets: Array of actual target values.
        :param predictions: Array of predicted values.
        :param delta: The threshold parameter that defines the point where the loss function changes.
        :return: A numpy array containing the gradients of the Huber loss function with respect to the predictions.
                 This array will have the same shape as the input predictions.
        """
        assert targets.size > 0 and predictions.size > 0, "The size of targets and predictions arrays must be bigger than 0."
        assert targets.size == predictions.size, "The size of targets and predictions arrays must be same."

        absolute_error = targets - predictions
        gradient = np.where(np.abs(absolute_error) <= delta,
                            absolute_error,  # Quadratic region
                            delta * np.sign(absolute_error))  # Linear region
        return gradient / predictions.shape[0]


class BinaryCrossEntropy(Loss):
    """
    Binary Cross Entropy (BCE) loss function.
    """
    def forward(self, targets: np.ndarray, predictions: np.ndarray, epsilon: float = 1e-15) -> float:
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
        assert targets.size > 0 and predictions.size > 0, "The size of targets and predictions arrays must be bigger than 0."
        assert targets.size == predictions.size, "The size of targets and predictions arrays must be same."
        assert np.all(np.isin(targets, [0, 1])), "Targets must contain only binary values (0 or 1)."
        assert np.all((predictions >= 0)) and np.all((predictions <= 1)), "Predictions must be in the range [0, 1]."

        predictions = np.clip(predictions, epsilon, 1 - epsilon)
        loss = -(targets * np.log(predictions) + (1 - targets) * np.log(1 - predictions)).mean()
        return loss

    def derivative(self, targets: np.ndarray, predictions: np.ndarray, epsilon: float = 1e-15) -> np.ndarray:
        """
        Calculates the gradient (derivative) of the Binary Cross-Entropy (BCE) loss function
        with respect to the predictions.
        :param targets: Array of actual binary target values (0 or 1).
        :param predictions: Array of predicted probabilities (between 0 and 1).
        :param epsilon: mall value to clip the predictions and avoid log(0). Default is 1e-15.
        :return: A numpy array containing the gradients of the BCE loss function with respect to the predictions.
                 This array will have the same shape as the input predictions.
        """
        assert targets.size > 0 and predictions.size > 0, "The size of targets and predictions arrays must be bigger than 0."
        assert targets.size == predictions.size, "The size of targets and predictions arrays must be same."

        predictions = np.clip(predictions, epsilon, 1 - epsilon)
        gradient = (1 - targets) / (1 - predictions) - (targets / predictions)
        return gradient / predictions.shape[0]


class CategoricalCrossEntropy(Loss):
    """
    Categorical Cross Entropy (CCE) loss function.
    """
    def forward(self, targets: np.ndarray, predictions: np.ndarray, epsilon: float = 1e-15) -> float:
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
        assert targets.size > 0 and predictions.size > 0, "The size of targets and predictions arrays must be bigger than 0."
        assert targets.size == predictions.size, "The size of targets and predictions arrays must be same."
        assert targets.ndim and predictions.ndim == 2, "Targets and predictions arrays must be 2-dimensional (one-hot encoded)."
        assert predictions.shape[1] == targets.shape[1], "The number of classes in targets and predictions must match."

        predictions = np.clip(predictions, epsilon, 1 - epsilon)
        loss = -np.sum(targets * np.log(predictions))
        return loss / predictions.shape[0]

    def derivative(self, targets: np.ndarray, predictions: np.ndarray, epsilon: float = 1e-15) -> np.ndarray:
        """
        Calculates the gradient (derivative) of the Categorical Cross-Entropy (CCE) loss function
        with respect to the predictions.
        :param targets: Array of actual target values in one-hot encoded format.
        :param predictions: Array of predicted probabilities for each class.
        :param epsilon: Small value to clip the predictions and avoid log(0). Default is 1e-15.
        :return: A numpy array containing the gradients of the CCE loss function with respect to the predictions.
                 This array will have the same shape as the input predictions.
        """
        assert targets.size > 0 and predictions.size > 0, "The size of targets and predictions arrays must be bigger than 0."
        assert targets.shape == predictions.shape, "The size of targets and predictions arrays must be the same."
        assert targets.ndim and predictions.ndim == 2, "Targets and predictions arrays must be 2-dimensional (one-hot encoded)."
        assert predictions.shape[1] == targets.shape[1], "The number of classes in targets and predictions must match."

        predictions = np.clip(predictions, epsilon, 1 - epsilon)
        gradient = -targets / predictions
        return gradient / predictions.shape[0]
