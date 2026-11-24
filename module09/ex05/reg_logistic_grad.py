import numpy as np


def vec_reg_logistic_grad(y, x, theta, lambda_):
    """Computes the regularized logistic gradient of three non-empty numpy.ndarray, without
    any for-loop. The three arrays must have compatible shapes.
    Args:
    y: has to be a numpy.ndarray, a vector of shape m * 1.
    x: has to be a numpy.ndarray, a matrix of shape m * n.
    theta: has to be a numpy.ndarray, a vector of shape n * 1.
    lambda_: has to be a float.
    Returns:
    A numpy.ndarray, a vector of shape n * 1, containing the results of the formula for all j.
    None if y, x, or theta are empty numpy.ndarray.
    None if y, x or theta does not share compatibles shapes.
    Raises:
    This function should not raise any Exception.
    """
    if (
        not isinstance(y, np.ndarray)
        or y.ndim != 2
        or y.shape[-1] != 1
        or y.size == 0
        or not isinstance(x, np.ndarray)
        or x.ndim != 2
        or x.size == 0
        or not isinstance(theta, np.ndarray)
        or theta.ndim != 2
        or theta.shape[-1] != 1
        or not isinstance(lambda_, float)
    ):
        return
    theta_reg = theta.copy()
    theta_reg[0] = 0
    X = np.hstack((np.ones((x.shape[0], 1)), x))
    return (
        X.T @ (1 / (1 + np.exp(-X @ theta)) - y) + lambda_ * theta_reg
    ) / X.shape[0]


# x = np.array([[0, 2, 3, 4],
# [2, 4, 5, 5],
# [1, 3, 2, 7]])
# y = np.array([[0], [1], [1]])
# theta = np.array([[-2.4], [-1.5], [0.3], [-1.4], [0.7]])
# # Example 1.1:
# print(vec_reg_logistic_grad(y, x, theta, 1.0))
# print(vec_reg_logistic_grad(y, x, theta, 0.5))
# print(vec_reg_logistic_grad(y, x, theta, 0.0))
