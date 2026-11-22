import numpy as np


class MyLinearRegression:
    """
    Description:
    My personnal linear regression class to fit like a boss.
    """

    def __init__(self, thetas, alpha=0.001, max_iter=1000):
        self.alpha = alpha
        self.max_iter = max_iter
        self.thetas = thetas

    def fit_(self, x, y):
        X = np.hstack((np.ones((x.shape[0], 1)), x))
        for _ in range(self.max_iter):
            graident = (X.T @ (X @ self.thetas - y)) / x.shape[0]
            self.thetas = self.thetas - graident * self.alpha

    def predict_(self, x):
        return self.thetas[0] + self.thetas[1] * x

    def loss_elem_(self, y, y_hat):
        return (y - y_hat) ** 2

    def loss_(self, y, y_hat):
        return np.mean(self.loss_elem_(y, y_hat)) / 2


x = np.array(
    [[12.4956442], [21.5007972], [31.5527382], [48.9145838], [57.5088733]]
)
y = np.array(
    [[37.4013816], [36.1473236], [45.7655287], [46.6793434], [59.5585554]]
)
lr1 = MyLinearRegression(np.array([[2], [0.7]]))
# Example 0.0:
y_hat = lr1.predict_(x)
print(y_hat)
print(lr1.loss_elem_(y, y_hat))

# Example 1.0:
lr2 = MyLinearRegression(np.array([[1], [1]]), 5e-8, 1500000)
lr2.fit_(x, y)
print(lr2.thetas)

y_hat = lr2.predict_(x)
print(y_hat)
print(lr2.loss_elem_(y, y_hat))
print(lr2.loss_(y, y_hat))
