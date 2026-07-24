'''
Implement Ridge and Lasso regression on the Diabetes dataset. Compare the performance
of these regularized models with standard linear regression.
Tasks:
● Load and preprocess the dataset.
● Implement Ridge and Lasso regression.
● Tune hyperparameters using cross-validation.
● Compare performance metrics (MSE, R-squared) with standard linear regression.
'''

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

linear = LinearRegression()
linear.fit(X_train, y_train)

ridge = RidgeCV(alphas=[0.01, 0.1, 1, 10, 100], cv=5)
ridge.fit(X_train, y_train)
lasso = LassoCV(alphas=[0.01, 0.1, 1, 10, 100], cv=5, random_state=42)
lasso.fit(X_train, y_train)

models = {
    "Linear Regression": linear,
    "Ridge Regression": ridge,
    "Lasso Regression": lasso
}

for name, model in models.items():

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(name)
    print("MSE :", mse)
    print("RMSE:", rmse)
    print("MAE :", mae)
    print("R2  :", r2)

    if name != "Linear Regression":
        print("Best Alpha:", model.alpha_)

    print()

plt.figure(figsize=(8,6))

plt.scatter(y_test, linear.predict(X_test), label="Linear Regression")
plt.scatter(y_test, ridge.predict(X_test), label="Ridge Regression")
plt.scatter(y_test, lasso.predict(X_test), label="Lasso Regression")

minimum = min(y_test)
maximum = max(y_test)

plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs Predicted")
plt.legend()
plt.grid(True)
plt.show()