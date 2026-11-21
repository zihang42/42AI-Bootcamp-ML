import numpy as np


def loss_(y, y_hat) -> None | np.ndarray:
    """Computes the half mean-squared-error of two non-empty numpy.arrays, without any for loop.
    The two arrays must have the same dimensions.
    Args:
    y: has to be an numpy.array, a one-dimensional array of size m.
    y_hat: has to be an numpy.array, a one-dimensional array of size m.
    Returns:
    The half mean-squared-error of the two vectors as a float.
    None if y or y_hat are empty numpy.array.
    None if y and y_hat does not share the same dimensions.
    Raises:
    This function should not raise any Exceptions.
    """

    if (
        not isinstance(y, np.ndarray)
        or not isinstance(y_hat, np.ndarray)
        or y.ndim != 1
        or y.shape == 0
        or y.shape != y_hat.shape
    ):
        return

    diff = y - y_hat
    return diff @ diff / (2 * y.size)


# X = np.array([0, 15, -9, 7, 12, 3, -21])
# Y = np.array([2, 14, -13, 5, 12, 4, -19])
# # Example 1:
# print(loss_(X, Y))
# print(loss_(X, X))
