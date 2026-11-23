import numpy as np


class MyLR:
    """
    Description:
    My personnal linear regression class to fit like a boss.
    """

    def __init__(self, thetas, alpha=0.001, max_iter=1000):
        self.alpha = alpha
        self.max_iter = max_iter
        self.thetas = np.asarray(thetas, dtype=float)

    def fit_(self, x, y):
        X = np.hstack((np.ones((x.shape[0], 1)), x))
        for _ in range(self.max_iter):
            graident = (X.T @ (X @ self.thetas - y)) / x.shape[0]
            self.thetas = self.thetas - graident * self.alpha

    def predict_(self, x):
        X = np.hstack((np.ones((x.shape[0], 1)), x))
        return X @ self.thetas

    def loss_elem_(self, y, y_hat):
        return (y - y_hat) ** 2

    def loss_(self, y, y_hat):
        return np.mean(self.loss_elem_(y, y_hat)) / 2


# X = np.array(
#     [[1.0, 1.0, 2.0, 3.0], [5.0, 8.0, 13.0, 21.0], [34.0, 55.0, 89.0, 144.0]]
# )
# Y = np.array([[23.0], [48.0], [218.0]])
# mylr = MyLR([[1.0], [1.0], [1.0], [1.0], [1]])
# # Example 0:
# y_hat = mylr.predict_(X)
# print(y_hat)

# # Example 1:
# print(mylr.loss_elem_(Y, y_hat))

# # Example 2:
# print(mylr.loss_(Y, y_hat))

# # Example 3:
# mylr.alpha = 1.6e-4
# mylr.max_iter = 200000
# mylr.fit_(X, Y)
# print(mylr.thetas)

# # Example 4:
# y_hat = mylr.predict_(X)
# print(y_hat)

# # Example 5:
# print(mylr.loss_elem_(Y, y_hat))

# # Example 6:
# print(mylr.loss_(Y, y_hat))
