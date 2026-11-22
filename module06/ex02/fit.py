import numpy as np


def fit_(
    x: np.ndarray,
    y: np.ndarray,
    theta: np.ndarray,
    alpha: float,
    max_iter: int,
):
    """
    Description:
    Fits the model to the training dataset contained in x and y.
    Args:
    x: has to be a numpy.ndarray, a vector of dimension m * 1: (number of training examples, 1).
    y: has to be a numpy.ndarray, a vector of dimension m * 1: (number of training examples, 1).
    theta: has to be a numpy.ndarray, a vector of dimension 2 * 1.
    alpha: has to be a float, the learning rate
    max_iter: has to be an int, the number of iterations done during the gradient descent
    Returns:
    new_theta: numpy.ndarray, a vector of dimension 2 * 1.
    None if there is a matching dimension problem.
    Raises:
    This function should not raise any Exception.
    """

    if (
        not isinstance(x, np.ndarray)
        or x.size == 0
        or x.ndim != 2
        or x.shape[-1] != 1
        or not isinstance(y, np.ndarray)
        or y.size == 0
        or y.ndim != 2
        or y.shape[-1] != 1
        or not isinstance(theta, np.ndarray)
        or theta.ndim != 2
        or theta.shape != (2, 1)
        or not isinstance(alpha, float)
        or not 0.0 < alpha < 1.0
        or not isinstance(max_iter, int)
        or max_iter < 0
    ):
        return
    X = np.hstack((np.ones((x.size, 1)), x))
    for _ in range(max_iter):
        gradient = (X.T @ (X @ theta - y)) / x.size
        theta = theta - alpha * gradient
    return theta


def predict(x: np.ndarray,  theta: np.ndarray):
    return theta[0, :] + theta[1, :] * x


# x = np.array(
#     [[12.4956442], [21.5007972], [31.5527382], [48.9145838], [57.5088733]]
# )
# y = np.array(
#     [[37.4013816], [36.1473236], [45.7655287], [46.6793434], [59.5585554]]
# )
# theta = np.array([1, 1]).reshape((-1, 1))
# # Example 0:
# theta1 = fit_(x, y, theta, alpha=5e-8, max_iter=1500000)
# print(theta1)


# # Example 1:
# print(predict(x, theta1))
