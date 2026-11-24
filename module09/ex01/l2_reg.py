import numpy as np


def iterative_l2(theta):
    """Computes the L2 regularization of a non-empty numpy.ndarray, with a for-loop.
    Args:
    theta: has to be a numpy.ndarray, a vector of shape n * 1.
    Returns:
    The L2 regularization as a float.
    None if theta in an empty numpy.ndarray.
    Raises:
    This function should not raise any Exception.
    """
    if (
        not isinstance(theta, np.ndarray)
        or theta.size == 0
        or theta.shape[-1] != 1
    ):
        return

    return float(np.sum(theta[1:]**2))


def l2(theta):
    """Computes the L2 regularization of a non-empty numpy.ndarray, without any for-loop.
    Args:
    theta: has to be a numpy.ndarray, a vector of shape n * 1.
    Returns:
    The L2 regularization as a float.
    None if theta in an empty numpy.ndarray.
    Raises:
    This function should not raise any Exception.
    """

    if (
        not isinstance(theta, np.ndarray)
        or theta.size == 0
        or theta.shape[-1] != 1
    ):
        return

    # Theta = np.hstack((np.zeros((theta.shape[0], 1)), theta))
    # return Theta.T @ Theta
    theta_copy = theta.copy()
    theta_copy[0] = 0
    return float((theta_copy.T @ theta_copy).item())


# x = np.array([2, 14, -13, 5, 12, 4, -19]).reshape((-1, 1))
# # Example 1:
# print(iterative_l2(x))

# # Example 2:
# print(l2(x))

# y = np.array([3,0.5,-6]).reshape((-1, 1))
# # Example 3:
# print(iterative_l2(y))

# # Example 4:
# print(l2(y))