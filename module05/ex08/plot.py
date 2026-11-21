import numpy as np
from matplotlib import pyplot as plt


def plot_with_loss(x: np.ndarray, y: np.ndarray, theta: np.ndarray):
    """Plot the data and prediction line from three non-empty numpy.ndarray.
    Args:
    x: has to be an numpy.ndarray, one-dimensional array of size m.
    y: has to be an numpy.ndarray, one-dimensional array of size m.
    theta: has to be an numpy.ndarray, one-dimensional array of size 2.
    Returns:
    Nothing.
    Raises:
    This function should not raise any Exception.
    """

    if (
        not isinstance(x, np.ndarray)
        or not isinstance(y, np.ndarray)
        or not isinstance(theta, np.ndarray)
        or x.ndim != 1
        or y.ndim != 1
        or x.size == 0
        or x.shape != y.shape
        or theta.shape != (2,)
    ):
        return
    y_hat = theta[0] + theta[1] * x
    diff = y - y_hat
    loss = diff @ diff / (y.size)

    plt.scatter(x, y)
    plt.plot(x, y_hat)
    for xi, yi, pi in zip(x, y, y_hat):
        plt.plot([xi, xi], [yi, pi], linestyle="--", color="gray", linewidth=1)
    plt.title(f"Cost: {loss:.6f}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()


# x = np.arange(1, 6)
# y = np.array([11.52434424, 10.62589482, 13.14755699, 18.60682298, 14.14329568])
# # Example 1:
# theta1 = np.array([18, -1])
# plot_with_loss(x, y, theta1)

# theta2 = np.array([14, 0])
# plot_with_loss(x, y, theta2)

# theta3 = np.array([12, 0.8])
# plot_with_loss(x, y, theta3)
