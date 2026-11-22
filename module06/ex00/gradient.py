import numpy as np


def simple_gradient(
    x: np.ndarray, y: np.ndarray, theta: np.ndarray
) -> None | np.ndarray:
    """Computes a gradient vector from three non-empty numpy.arrays, with a for-loop.
    The three arrays must have compatible shapes.
    Args:
    x: has to be an numpy.array, a vector of shape m * 1.
    y: has to be an numpy.array, a vector of shape m * 1.
    theta: has to be an numpy.array, a 2 * 1 vector.
    Return:
    The gradient as a numpy.array, a vector of shape 2 * 1.
    None if x, y, or theta are empty numpy.array.
    None if x, y and theta do not have compatible shapes.
    None if x, y or theta is not of the expected type.
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
    ):
        return

    prediction = theta[0] + theta[1] * x
    nabla0 = np.mean(prediction - y)
    nabla1 = np.mean((prediction - y) * x)
    return np.vstack((nabla0, nabla1))


# x = np.array(
#     [12.4956442, 21.5007972, 31.5527382, 48.9145838, 57.5088733]
# ).reshape((-1, 1))
# y = np.array(
#     [37.4013816, 36.1473236, 45.7655287, 46.6793434, 59.5585554]
# ).reshape((-1, 1))
# # Example 0:
# theta1 = np.array([2, 0.7]).reshape((-1, 1))
# print(simple_gradient(x, y, theta1))

# # Example 1:
# theta2 = np.array([1, -0.4]).reshape((-1, 1))
# print(simple_gradient(x, y, theta2))
