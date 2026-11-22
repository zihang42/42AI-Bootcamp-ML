import matplotlib.pyplot as plt
import numpy as np


class MyLR:
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

    def mse_(self, y, y_hat):
        return np.mean(self.loss_elem_(y, y_hat))

    def plot_prediction(self, x, y, y_hat):
        plt.scatter(x, y, label="true")
        plt.scatter(x, y_hat, marker="x", label="predict")
        plt.plot(x, y_hat, linestyle="dashed")
        plt.xlabel("Quantity of blue pills (in micrograms)")
        plt.ylabel("Space driving score")
        plt.grid()
        plt.legend()
        plt.show()


# data = pd.read_csv("module06/ex04/are_blue_pills_magics.csv")
# Xpill = np.array(data["Micrograms"]).reshape(-1, 1)
# Yscore = np.array(data["Score"]).reshape(-1, 1)
# linear_model1 = MyLR(np.array([[89.0], [-8]]))
# linear_model2 = MyLR(np.array([[89.0], [-6]]))

# Y_model1 = linear_model1.predict_(Xpill)
# Y_model2 = linear_model2.predict_(Xpill)
# print(linear_model1.mse_(Yscore, Y_model1))
# # 57.60304285714282
# print(mean_squared_error(Yscore, Y_model1))
# # 57.603042857142825
# print(linear_model2.mse_(Yscore, Y_model2))
# # 232.16344285714285
# print(mean_squared_error(Yscore, Y_model2))
# # 232.16344285714285
# linear_model1.fit_(Xpill, Yscore)
# Yl1 = linear_model1.predict_(Xpill)

# linear_model1.plot_prediction(Xpill, Yscore, Yl1)
