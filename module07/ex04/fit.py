import numpy as np


def fit_(x, y, theta, alpha, max_iter):
    """
    Description:
    Args:
    Fits the model to the training dataset contained in x and y.
    x: has to be a numpy.array, a matrix of dimension m * n:
    (number of training examples, number of features).
    y: has to be a numpy.array, a vector of dimension m * 1:
    (number of training examples, 1).
    theta: has to be a numpy.array, a vector of dimension (n + 1) * 1:
    (number of features + 1, 1).
    alpha: has to be a float, the learning rate
    max_iter: has to be an int, the number of iterations done during the gradient descent
    Return:
    new_theta: numpy.array, a vector of dimension (number of features + 1, 1).
    None if there is a matching dimension problem.
    None if x, y, theta, alpha or max_iter is not of expected type.
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
        or alpha <= 0.0
        or not isinstance(max_iter, int)
    ):
        return
    for _ in range(max_iter):
        X = np.hstack((np.ones((x.shape[0], 1)), x))
        gradient = X.T @ (X @ theta - y) / X.shape[0]
        theta = theta - alpha * gradient
    return theta


# x = np.array(
#     [[0.2, 2.0, 20.0], [0.4, 4.0, 40.0], [0.6, 6.0, 60.0], [0.8, 8.0, 80.0]]
# )
# y = np.array([[19.6], [-2.8], [-25.2], [-47.6]])
# theta = np.array([[42.0], [1.0], [1.0], [1.0]])
# # Example 0:
# theta2 = fit_(x, y, theta, alpha=0.0005, max_iter=42000)
# print(theta2)


# # Example 1:
# def predict_(x: np.ndarray, theta: np.ndarray):
#     """Computes the prediction vector y_hat from two non-empty numpy.array.
#     Args:
#     x: has to be an numpy.array, a matrix of dimension m * n.
#     theta: has to be an numpy.array, a vector of dimension (n + 1) * 1.
#     Return:
#     y_hat as a numpy.array, a vector of dimension m * 1.
#     None if x or theta are empty numpy.array.
#     None if x or theta dimensions are not matching.
#     None if x or theta is not of expected type.
#     Raises:
#     This function should not raise any Exception.
#     """

#     if (
#         not isinstance(x, np.ndarray)
#         or x.size == 0
#         or x.ndim != 2
#         or not isinstance(theta, np.ndarray)
#         or theta.size == 0
#         or theta.ndim != 2
#     ):
#         return
#     X = np.hstack((np.ones((x.shape[0], 1)), x))
#     return X @ theta


# print(predict_(x, theta2))
