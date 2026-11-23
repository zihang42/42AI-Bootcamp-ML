import numpy as np


def data_spliter(x, y, proportion):
    """Shuffles and splits the dataset (given by x and y) into a training and a test set,
    while respecting the given proportion of examples to be kept in the training set.
    Args:
    x: has to be an numpy.array, a matrix of dimension m * n.
    y: has to be an numpy.array, a vector of dimension m * 1.
    proportion: has to be a float, the proportion of the dataset that will be assigned to the
    training set.
    Return:
    (x_train, x_test, y_train, y_test) as a tuple of numpy.array
    None if x or y is an empty numpy.array.
    None if x and y do not share compatible dimensions.
    None if x, y or proportion is not of expected type.
    Raises:
    This function should not raise any Exception.
    """
    if (
        not isinstance(x, np.ndarray)
        or not isinstance(y, np.ndarray)
        or not isinstance(proportion, float)
        or x.ndim != 2
        or y.ndim != 2
        or y.shape[1] != 1
        or x.size == 0
        or y.size == 0
        or x.shape[0] != y.shape[0]
        or not 0 <= proportion <= 1
    ):
        return None
    m = x.shape[0]

    np.random.seed(42)
    indices = np.random.permutation(m)

    x_shuffled = x[indices]
    y_shuffled = y[indices]

    split_index = int(m * proportion)

    x_train = x_shuffled[:split_index]
    x_test = x_shuffled[split_index:]

    y_train = y_shuffled[:split_index]
    y_test = y_shuffled[split_index:]

    return x_train, x_test, y_train, y_test
