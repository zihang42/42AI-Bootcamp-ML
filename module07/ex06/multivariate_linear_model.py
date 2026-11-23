import matplotlib.pyplot as plt
import numpy as np



class UnivariateLR:
    """
    Description:
    My personnal linear regression class to fit like a boss.
    """

    def __init__(self, thetas, alpha=0.001, max_iter=1000):
        self.alpha = alpha
        self.max_iter = max_iter
        self.thetas = np.array(thetas, dtype=float).reshape(-1, 1)

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

    def mse_(self, y, y_hat):
        return np.mean(self.loss_elem_(y, y_hat))

    def plot_prediction(self, x, y, y_hat):
        plt.scatter(x, y, label="true")
        plt.scatter(x, y_hat, marker="x", label="predict")
        plt.plot(x, y_hat, linestyle="dashed")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.grid()
        plt.legend()
        plt.show()

# import pandas as pd
# data = pd.read_csv("module07/ex06/spacecraft_data.csv")
# X = np.array(data[["Age"]])
# Y = np.array(data[["Sell_price"]])
# myLR_age = UnivariateLR(thetas=[[1000.0], [-1.0]], alpha=2.5e-5, max_iter=100000)
# myLR_age.fit_(X, Y)
# y_pred = myLR_age.predict_(X)
# print(myLR_age.mse_(Y, y_pred))

# for c in data.iloc[:, :-1].columns:
#     X = data[[c]].to_numpy()
#     myLR_age = UnivariateLR(thetas=[[1000.0], [-1.0]], alpha=2.5e-5, max_iter=100000)
#     myLR_age.fit_(X, Y)
#     y_pred = myLR_age.predict_(X)
#     myLR_age.plot_prediction(X, Y, y_pred)


# data = pd.read_csv("module07/ex06/spacecraft_data.csv")
# X = np.array(data[['Age','Thrust_power','Terameters']])
# Y = np.array(data[['Sell_price']])
# my_lreg = UnivariateLR(thetas=[1.0, 1.0, 1.0, 1.0], alpha=9e-5, max_iter=500000)
# # Example 0:
# print(my_lreg.mse_(Y, my_lreg.predict_(X)))


# # Example 1:
# my_lreg.fit_(X, Y)
# print(my_lreg.thetas)


# # Example 2:
# print(my_lreg.mse_(Y, my_lreg.predict_(X)))