import pickle
from itertools import combinations

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

FEATURES = ["weight", "prod_distance", "time_delivery"]

data = pd.read_csv("space_avocado.csv")

X = data[FEATURES]
y = data["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

models = {}
results = []

for size in range(1, len(FEATURES) + 1):
    for features in combinations(FEATURES, size):
        for degree in range(1, 5):
            model = make_pipeline(
                PolynomialFeatures(degree=degree, include_bias=False),
                LinearRegression(),
            )

            model.fit(X_train[list(features)], y_train)

            y_pred = model.predict(X_test[list(features)])
            mse = mean_squared_error(y_test, y_pred)

            name = f"{'_'.join(features)}_degree_{degree}"

            models[name] = {
                "features": list(features),
                "degree": degree,
                "model": model,
                "mse": float(mse),
            }

            results.append({"name": name, "mse": mse})

            print(f"{name}: MSE = {mse:.4f}")

results.sort(key=lambda item: item["mse"])
best_name = results[0]["name"]

print("\nBest model:", best_name)
print("Best test MSE:", results[0]["mse"])

with open("models.pkl", "wb") as file:
    pickle.dump({"models": models, "best_name": best_name}, file)
