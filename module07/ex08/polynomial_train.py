import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures

data = pd.read_csv("module07/ex08/are_blue_pills_magics.csv")

X = data[["Micrograms"]].to_numpy()
y = data["Score"].to_numpy()

models = {}
mse_scores = []
X_plot = np.linspace(X.min(), X.max(), 500).reshape(-1, 1)
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color="black", label="Data", zorder=5)

for degree in range(1, 7):
    poly = PolynomialFeatures(degree=degree, include_bias=False)

    X_poly = poly.fit_transform(X)

    model = LinearRegression()
    model.fit(X_poly, y)

    y_pred = model.predict(X_poly)

    mse = mean_squared_error(y, y_pred)
    mse_scores.append(mse)

    print(f"Degree {degree}: MSE = {mse:.6f}")

    X_plot_poly = poly.transform(X_plot)
    y_plot = model.predict(X_plot_poly)

    plt.plot(X_plot, y_plot, label=f"Degree {degree}")

    models[degree] = (poly, model)

plt.xlabel("Micrograms")
plt.ylabel("Score")
plt.title("Polynomial Regression (Degrees 1-6)")
plt.legend()
plt.grid(alpha=0.3)
plt.show()

plt.figure(figsize=(8, 5))

plt.bar(range(1, 7), mse_scores)

plt.xlabel("Polynomial Degree")
plt.ylabel("Mean Squared Error")
plt.title("MSE by Polynomial Degree")
plt.xticks(range(1, 7))
plt.grid(axis="y", alpha=0.3)

plt.show()
