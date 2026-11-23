import numpy as np


def simple_predict(x: np.ndarray, theta: np.ndarray):
    """Computes the prediction vector y_hat from two non-empty numpy.array.
    Args:
    x: has to be an numpy.array, a matrix of dimension m * n.
    theta: has to be an numpy.array, a vector of dimension (n + 1) * 1.
    Return:
    y_hat as a numpy.array, a vector of dimension m * 1.
    None if x or theta are empty numpy.array.
    None if x or theta dimensions are not matching.
    None if x or theta is not of expected type.
    Raises:
    This function should not raise any Exception.
    """

    if (
        not isinstance(x, np.ndarray)
        or x.size == 0
        or x.ndim != 2
        or not isinstance(theta, np.ndarray)
        or theta.size == 0
        or theta.ndim != 2
    ):
        return
    X = np.hstack((np.ones((x.shape[0], 1)), x))
    return X @ theta


# x = np.arange(1, 13).reshape((4, -1))
# # Example 1:
# theta1 = np.array([5, 0, 0, 0]).reshape((-1, 1))
# print(simple_predict(x, theta1))

# # Example 2:
# theta2 = np.array([0, 1, 0, 0]).reshape((-1, 1))
# print(simple_predict(x, theta2))


# # Example 3:
# theta3 = np.array([-1.5, 0.6, 2.3, 1.98]).reshape((-1, 1))
# print(simple_predict(x, theta3))


# # Example 4:
# theta4 = np.array([-3, 1, 2, 3.5]).reshape((-1, 1))
# print(simple_predict(x, theta4))
