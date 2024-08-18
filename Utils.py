import numpy as np
from typing import Union, List, Tuple
import pandas as pd


def one_hot_encode(array: Union[List, np.ndarray, pd.DataFrame]) -> Tuple[np.ndarray, List[str]]:
    """
    Converts a given array of integers to a one-hot encoded matrix and returns the class names.
    :param array: (list, np.ndarray, or pd.DataFrame) A list, numpy array, or pandas DataFrame of integers to be one-hot encoded.
    :return: A tuple containing:
             - A one-hot encoded 2D numpy array.
             - A list of class names (integers) corresponding to the columns of the one-hot encoded matrix.
    """
    if isinstance(array, pd.DataFrame):
        array = array.values.flatten()
    elif isinstance(array, list):
        array = np.array(array)
    elif isinstance(array, np.ndarray) and array.ndim > 1:
        array = array.flatten()

    classes = np.unique(array)
    class_to_index = {cls: i for i, cls in enumerate(classes)}

    one_hot_matrix = np.zeros((array.size, classes.size))
    for i, value in enumerate(array):
        if isinstance(value, np.ndarray):
            value = value.item()
        one_hot_matrix[i, class_to_index[value]] = 1

    return one_hot_matrix, [f'class_{cls}' for cls in classes]

def split_train_test(data: Union[List, np.ndarray, pd.DataFrame], split_ratio: float = 0.2, shuffle: bool = True) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Splits the given dataset into training and testing sets.
    :param data: (list or np.ndarray) The dataset to be split. It can be a Python list or a NumPy array.
    :param split_ratio: (float) The proportion of the dataset to be used as the test set. Should be a value between 0.0 and 1.0.
    :param shuffle: If True, the data will be shuffled before splitting. Defaults to True.
    :return:
    """
    assert 0.0 < split_ratio < 1.0, "Please select a correct split ratio between 0.0 and 1.0"
    if isinstance(data, list):
        data = np.array(data)
    if shuffle:
        if isinstance(data, pd.DataFrame):
            data = data.sample(frac=1).reset_index(drop=True)
        else:
            np.random.shuffle(data)

    test_length = int(len(data) * split_ratio)
    test = data[:test_length]
    train = data[test_length:]
    return train, test
