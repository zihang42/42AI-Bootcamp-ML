import numpy as np


def log_gradient(x, y, theta):
    """Computes a gradient vector from three non-empty numpy.ndarray, with a for-loop. The three arrays must have compatibl
    Args:
    x: has to be an numpy.ndarray, a matrix of shape m * n.
    y: has to be an numpy.ndarray, a vector of shape m * 1.
    theta: has to be an numpy.ndarray, a vector of shape (n + 1) * 1.
    Returns:
    The gradient as a numpy.ndarray, a vector of shape n * 1, containing the result of the formula for all j.
    None if x, y, or theta are empty numpy.ndarray.
    None if x, y and theta do not have compatible dimensions.
    Raises:
    This function should not raise any Exception.
    """

    if (
        not isinstance(x, np.ndarray)
        or x.size == 0
        or x.ndim != 2
        or not isinstance(y, np.ndarray)
        or y.size == 0
        or y.ndim != 2
        or y.shape[-1] != 1
        or not isinstance(theta, np.ndarray)
        or theta.ndim != 2
    ):
        return

    X = np.hstack((np.ones((x.shape[0], 1)), x))
    return X.T @ ((1 / (1 + np.e ** (-X @ theta))) - y) / x.shape[0]
