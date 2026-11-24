import numpy as np


def sigmoid_(x):
    """
    Compute the sigmoid of a vector.
    Args:
    x: has to be a numpy.ndarray of shape (m, 1).
    Returns:
    The sigmoid value as a numpy.ndarray of shape (m, 1).
    None if x is an empty numpy.ndarray.
    Raises:
    This function should not raise any Exception.
    """
    if (
        not isinstance(x, np.ndarray)
        or x.size == 0
        or x.ndim != 2
        or x.shape[-1] != 1
    ):
        return

    return 1 / (1 + np.e ** (-x))


# # Example 1:
# x = np.array([[-4]])
# print(sigmoid_(x))

# # Example 2:
# x = np.array([[2]])
# print(sigmoid_(x))

# # Example 3:
# x = np.array([[-4], [2], [0]])
# print(sigmoid_(x))