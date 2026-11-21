import numpy as np


def predict_(x: np.ndarray, theta: np.ndarray):
    """Computes the vector of prediction y_hat from two non-empty numpy.array.
    Args:
    x: has to be an numpy.array, a one-dimensional array of size m.
    theta: has to be an numpy.array, a two-dimensional array of shape 2 * 1.
    Returns:
    y_hat as a numpy.array, a two-dimensional array of shape m * 1.
    None if x and/or theta are not numpy.array.
    None if x or theta are empty numpy.array.
    None if x or theta dimensions are not appropriate.
    Raises:
    This function should not raise any Exceptions.
    """

    if (
        not isinstance(x, np.ndarray)
        or not isinstance(theta, np.ndarray)
        or theta.shape != (2, 1)
        or x.ndim != 2
        or x.size == 0
        or x.shape[-1] != 1
        or theta.ndim != 2
    ):
        return

    matrix = np.hstack((np.ones((x.shape[0], 1)), x))

    return matrix @ theta


# x = np.arange(1, 6)
# theta1 = np.array([[5], [0]])
# print(predict_(x, theta1))

# theta2 = np.array([[0], [1]])
# print(predict_(x, theta2))

# theta3 = np.array([[5], [3]])
# print(predict_(x, theta3))


# theta4 = np.array([[-3], [1]])
# print(predict_(x, theta4))
