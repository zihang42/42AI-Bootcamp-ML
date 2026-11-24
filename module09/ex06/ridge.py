import numpy as np


class MyLinearRegression:
    def __init__(self, thetas, alpha=0.001, max_iter=1000):
        self.alpha = alpha
        self.max_iter = max_iter
        self.thetas = np.array(thetas, dtype=float).reshape(-1, 1)

    def predict_(self, x):
        if not isinstance(x, np.ndarray) or x.size == 0:
            return None
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        if x.shape[1] + 1 != self.thetas.shape[0]:
            return None
        x_prime = np.hstack((np.ones((x.shape[0], 1)), x))
        return x_prime @ self.thetas

    def loss_elem_(self, y, y_hat):
        if not isinstance(y, np.ndarray) or not isinstance(y_hat, np.ndarray):
            return None
        if y.shape != y_hat.shape or y.size == 0:
            return None
        return (y_hat - y) ** 2


class MyRidge(MyLinearRegression):
    def __init__(self, thetas, alpha=0.001, max_iter=1000, lambda_=0.5):
        super().__init__(thetas, alpha, max_iter)
        self.lambda_ = lambda_

    def get_params_(self):
        return {
            "thetas": self.thetas,
            "alpha": self.alpha,
            "max_iter": self.max_iter,
            "lambda_": self.lambda_,
        }

    def set_params_(self, **params):
        for key, value in params.items():
            if hasattr(self, key):
                if key == "thetas":
                    value = np.array(value, dtype=float).reshape(-1, 1)
                setattr(self, key, value)
        return self

    def loss_elem_(self, y, y_hat):
        return super().loss_elem_(y, y_hat)

    def loss_(self, y, y_hat):
        if not isinstance(y, np.ndarray) or not isinstance(y_hat, np.ndarray):
            return None
        if y.shape != y_hat.shape or y.size == 0:
            return None

        m = y.shape[0]
        theta_prime = self.thetas.copy()
        theta_prime[0] = 0.0

        mse_loss = float(np.sum((y_hat - y) ** 2))
        l2_reg = float(self.lambda_ * np.sum(theta_prime**2))
        return (mse_loss + l2_reg) / (2 * m)

    def gradient_(self, x, y):
        if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray):
            return None
        if x.size == 0 or y.size == 0:
            return None
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        if x.shape[0] != y.shape[0] or x.shape[1] + 1 != self.thetas.shape[0]:
            return None

        m = x.shape[0]
        x_prime = np.hstack((np.ones((m, 1)), x))

        theta_prime = self.thetas.copy()
        theta_prime[0] = 0.0

        grad = (
            x_prime.T @ (x_prime @ self.thetas - y)
            + self.lambda_ * theta_prime
        ) / m
        return grad

    def fit_(self, x, y):
        if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray):
            return None
        if x.size == 0 or y.size == 0:
            return None
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        if x.shape[0] != y.shape[0] or x.shape[1] + 1 != self.thetas.shape[0]:
            return None

        for _ in range(self.max_iter):
            grad = self.gradient_(x, y)
            if grad is None:
                return None
            self.thetas = self.thetas - self.alpha * grad
        return self.thetas
