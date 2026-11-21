import matplotlib.pyplot as plt
import numpy as np


def plot(x: np.ndarray, y: np.ndarray, theta: np.ndarray):
    """Plot the data and prediction line from three non-empty numpy.array.
    Args:
    x: has to be an numpy.array, a one-dimensional array of size m.
    y: has to be an numpy.array, a one-dimensional array of size m.
    theta: has to be an numpy.array, a two-dimensional array of shape 2 * 1.
    Returns:
    Nothing.
    Raises:
    This function should not raise any Exceptions.
    """

    if (
        not isinstance(x, np.ndarray)
        or x.ndim != 1
        or x.size == 0
        or not isinstance(y, np.ndarray)
        or y.ndim != 1
        or not isinstance(theta, np.ndarray)
        or theta.shape != (2, 1)
    ):
        return

    prediction = theta[0, :] + theta[1, :] * x
    plt.scatter(x, y)
    plt.plot(x, prediction, label="Prediction")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.show()


# x = np.arange(1, 6)
# y = np.array([3.74013816, 3.61473236, 4.57655287, 4.66793434, 5.95585554])
# # Example 1:
# theta1 = np.array([[4.5], [-0.2]])
# plot(x, y, theta1)
