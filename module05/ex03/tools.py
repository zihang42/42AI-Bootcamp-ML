import numpy as np


def add_intercept(x: np.ndarray) -> None | np.ndarray:
    """Adds a column of 1’s to the non-empty numpy.array x.
    Args:
    x: has to be a numpy.array. x can be a one-dimensional (m * 1) or two-dimensional (m * n) array.
    Returns:
    X, a numpy.array of dimension m * (n + 1).
    None if x is not a numpy.array.
    None if x is an empty numpy.array.
    Raises:
    This function should not raise any Exception.
    """
    if not isinstance(x, np.ndarray) or x.ndim not in (1, 2) or x.size == 0:
        return

    matrix = x.reshape(-1, 1) if x.ndim == 1 else x
    return np.hstack((np.ones((matrix.shape[0], 1)), matrix)).astype(float)


# x = np.arange(1, 6)
# print(add_intercept(x))

# y = np.arange(1,10).reshape((3,3))
# print(add_intercept(y))