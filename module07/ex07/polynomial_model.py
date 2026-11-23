import numpy as np


def add_polynomial_features(x, power):
    """Add polynomial features to vector x by raising its values up to the power given in argument.
    Args:
    x: has to be an numpy.array, a vector of dimension m * 1.
    power: has to be an int, the power up to which the components of vector x are going to be raised.
    Return:
    The matrix of polynomial features as a numpy.array, of dimension m * n,
    containing the polynomial feature values for all training examples.
    None if x is an empty numpy.array.
    None if x or power is not of expected type.
    Raises:
    This function should not raise any Exception.
    """
    if (
        not isinstance(x, np.ndarray)
        or x.size == 0
        or not isinstance(power, int)
    ):
        return

    result = x.copy()

    for i in range(2, power + 1):
        result = np.hstack((result, x ** i))

    return result

# x = np.arange(1,6).reshape(-1, 1)
# # Example 0:
# print(add_polynomial_features(x, 3))

# # Example 1:
# print(add_polynomial_features(x, 6))