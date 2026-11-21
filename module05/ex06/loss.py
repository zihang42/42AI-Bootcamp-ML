import numpy as np


def loss_elem_(y: np.ndarray, y_hat: np.ndarray) -> None | np.ndarray:
    """
    Description:
    Calculates all the elements (y_pred - y)^2 of the loss function.
    Args:
    y: has to be an numpy.array, a two-dimensional array of shape m * 1.
    y_hat: has to be an numpy.array, a two-dimensional array of shape m * 1.
    Returns:
    J_elem: numpy.array, a array of dimension (number of the training examples, 1).
    None if there is a dimension matching problem.
    None if any argument is not of the expected type.
    Raises:
    This function should not raise any Exception.
    """
    if (
        not isinstance(y, np.ndarray)
        or not isinstance(y_hat, np.ndarray)
        or y.ndim != 2
        or y.shape[-1] != 1
        or y.size == 0
        or not (y_hat, np.ndarray)
        or y_hat.shape != y.shape
    ):
        return

    return (y - y_hat) ** 2


def loss_(y: np.ndarray, y_hat: np.ndarray) -> float | None:
    """
    Description:
    Calculates the value of loss function.
    Args:
    y: has to be an numpy.array, a two-dimensional array of shape m * 1.
    y_hat: has to be an numpy.array, a two-dimensional array of shape m * 1.
    Returns:
    J_value : has to be a float.
    None if there is a dimension matching problem.
    None if any argument is not of the expected type.
    Raises:
    This function should not raise any Exception.
    """
    loss_matrix = loss_elem_(y, y_hat)
    if loss_matrix is None:
        return
    return float(np.sum(loss_matrix) / (2 * y.size))


# def predict_(x: np.ndarray, theta: np.ndarray):
#     if (
#         not isinstance(x, np.ndarray)
#         or not isinstance(theta, np.ndarray)
#         or x.ndim != 2
#         or x.shape[1] != 1
#         or x.size == 0
#         or theta.shape != (2, 1)
#     ):
#         return None

#     matrix = np.hstack((np.ones((x.shape[0], 1)), x))

#     return matrix @ theta


# x1 = np.array([[0.0], [1.0], [2.0], [3.0], [4.0]])
# theta1 = np.array([[2.0], [4.0]])
# y_hat1 = predict_(x1, theta1)

# y1 = np.array([[2.0], [7.0], [12.0], [17.0], [22.0]])
# # Example 1:
# print(loss_elem_(y1, y_hat1))
# print(loss_(y1, y_hat1))
# x2 = np.array([0, 15, -9, 7, 12, 3, -21]).reshape(-1, 1)
# theta2 = np.array(np.array([[0.], [1.]]))
# y_hat2 = predict_(x2, theta2)
# y2 = np.array([2, 14, -13, 5, 12, 4, -19]).reshape(-1, 1)
# print(loss_(y2, y_hat2))
# print(loss_(y2, y2))
