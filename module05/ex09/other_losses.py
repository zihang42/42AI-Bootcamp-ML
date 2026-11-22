import numpy as np


def mse_(y: np.ndarray, y_hat: np.ndarray) -> None | np.float64:
    """
    Description:
    Calculate the MSE between the predicted output and the real output.
    Args:
    y: has to be a numpy.array, a two-dimensional array of shape m * 1.
    y_hat: has to be a numpy.array, a two-dimensional vector of shape m * 1.
    Returns:
    mse: has to be a float.
    None if there is a matching dimension problem.
    Raises:
    This function should not raise any Exceptions.
    """
    if (
        not isinstance(y, np.ndarray)
        or not isinstance(y_hat, np.ndarray)
        or y.ndim != 2
        or y.shape[1] != 1
        or y.shape == 0
        or y.shape != y_hat.shape
    ):
        return

    return np.mean((y - y_hat) ** 2)


def rmse_(y: np.ndarray, y_hat: np.ndarray) -> None | np.float64:
    """
    Description:
    Calculate the RMSE between the predicted output and the real output.
    Args:
    y: has to be a numpy.array, a two-dimensional array of shape m * 1.
    y_hat: has to be a numpy.array, a two-dimensional array of shape m * 1.
    Returns:
    rmse: has to be a float.
    None if there is a matching dimension problem.
    Raises:
    This function should not raise any Exceptions.
    """
    mse = mse_(y, y_hat)
    if mse is None:
        return
    return np.sqrt(mse)


def mae_(y: np.ndarray, y_hat: np.ndarray) -> None | np.float64:
    """
    Description:
    Calculate the MAE between the predicted output and the real output.
    Args:
    y: has to be a numpy.array, a two-dimensional array of shape m * 1.
    y_hat: has to be a numpy.array, a two-dimensional array of shape m * 1.
    Returns:
    mae: has to be a float.
    None if there is a matching dimension problem.
    Raises:
    This function should not raise any Exceptions.
    """
    if (
        not isinstance(y, np.ndarray)
        or not isinstance(y_hat, np.ndarray)
        or y.ndim != 2
        or y.shape[1] != 1
        or y.shape == 0
        or y.shape != y_hat.shape
    ):
        return
    return np.mean(np.abs(y - y_hat))


def r2score_(y: np.ndarray, y_hat: np.ndarray) -> None | np.float64:
    """
    Description:
    Calculate the R2score between the predicted output and the output.
    Args:
    y: has to be a numpy.array, a two-dimensional array of shape m * 1.
    y_hat: has to be a numpy.array, a two-dimensional array of shape m * 1.
    Returns:
    r2score: has to be a float.
    None if there is a matching dimension problem.
    Raises:
    This function should not raise any Exceptions.
    """
    mse = mse_(y, y_hat)

    var = np.var(y)
    if mse is None or var == 0.0:
        return
    return 1 - mse / var
