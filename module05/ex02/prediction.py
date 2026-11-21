import numpy as np


def simple_predict(x: np.ndarray, theta: np.ndarray) -> np.ndarray | None:
    if (
        not isinstance(x, np.ndarray)
        or not isinstance(theta, np.ndarray)
        or x is None
        or theta is None
        or x.ndim != 1
        or theta.ndim != 1
        or theta.size != 2
    ):
        return

    return (theta[0] + theta[1] * x).astype(float)


# x = np.arange(1,6)
# # Example 1:
# theta1 = np.array([5, 0])
# print(simple_predict(x, theta1))

# theta2 = np.array([0, 1])
# print(simple_predict(x, theta2))
# # Output:
# # array([1., 2., 3., 4., 5.])
# # Do you understand why y_hat == x here?
# # Example 3:
# theta3 = np.array([5, 3])
# print(simple_predict(x, theta3))
# # Output:
# # array([ 8., 11., 14., 17., 20.])
# # Example 4:
# theta4 = np.array([-3, 1])
# print(simple_predict(x, theta4))
# # Output:
# # array([-2., -1., 0., 1., 2.])
