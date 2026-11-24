import numpy as np


def reg_log_loss_(y, y_hat, theta, lambda_):
    """Computes the regularized loss of a linear regression model from two non-empty numpy.array,
    without any for loop. The two arrays must have the same dimensions.
    Args:
    y: has to be an numpy.ndarray, a vector of shape m * 1.
    y_hat: has to be an numpy.ndarray, a vector of shape m * 1.
    theta: has to be a numpy.ndarray, a vector of shape n * 1.
    lambda_: has to be a float.
    Returns:
    The regularized loss as a float.
    None if y, y_hat, or theta are empty numpy.ndarray.
    None if y and y_hat do not share the same shapes.
    Raises:
    This function should not raise any Exception.
    """

    if (
        not isinstance(y, np.ndarray)
        or y.ndim != 2
        or y.shape[-1] != 1
        or y.size == 0
        or not isinstance(y_hat, np.ndarray)
        or y_hat.ndim != 2
        or y_hat.shape != y.shape
        or not isinstance(theta, np.ndarray)
        or theta.ndim != 2
        or theta.shape[-1] != 1
        or not isinstance(lambda_, float)
    ):
        return
    theta[0] = 0
    return (
        (y.T@ np.log(y_hat) + (1 - y).T @ np.log(1 - y_hat) ) / -y.shape[0] + lambda_/(2 * y.shape[0]) * (theta.T @ theta)
    ).item() 


# y = np.array([1, 1, 0, 0, 1, 1, 0]).reshape((-1, 1))
# y_hat = np.array([.9, .79, .12, .04, .89, .93, .01]).reshape((-1, 1))
# theta = np.array([1, 2.5, 1.5, -0.9]).reshape((-1, 1))
# # Example :
# print(reg_log_loss_(y, y_hat, theta, .5))
# # Example :
# print(reg_log_loss_(y, y_hat, theta, .05))
# # Example :
# print(reg_log_loss_(y, y_hat, theta, .9))