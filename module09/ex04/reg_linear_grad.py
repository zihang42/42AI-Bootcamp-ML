import numpy as np


def vec_reg_linear_grad(y, x, theta, lambda_):
    """Computes the regularized linear gradient of three non-empty numpy.ndarray,
    without any for-loop. The three arrays must have compatible shapes.
    Args:
    y: has to be a numpy.ndarray, a vector of shape m * 1.
    x: has to be a numpy.ndarray, a matrix of dimesion m * n.
    theta: has to be a numpy.ndarray, a vector of shape (n + 1) * 1.
    lambda_: has to be a float.
    Return:
    A numpy.ndarray, a vector of shape (n + 1) * 1, containing the results of the formula for all j.
    None if y, x, or theta are empty numpy.ndarray.
    None if y, x or theta does not share compatibles shapes.
    None if y, x or theta or lambda_ is not of the expected type.
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

    X = np.hstack((np.ones((x.shape[0], 1)), x))
    return (X.T @ (X @ theta - y) + lambda_ * theta) / X.shape[0]


x = np.array([
[ -6, -7, -9],
[ 13, -2, 14],
[ -7, 14, -1],
[ -8, -4, 6],
[ -5, -9, 6],
[ 1, -5, 11],
[ 9, -11, 8]])
y = np.array([[2], [14], [-13], [5], [12], [4], [-19]])
theta = np.array([[7.01], [3], [10.5], [-6]])
# Example 1.1:
print(vec_reg_linear_grad(y, x, theta, 1.0))

# Example 2.2:
print(vec_reg_linear_grad(y, x, theta, 0.5))

print(vec_reg_linear_grad(y, x, theta, 0.0))