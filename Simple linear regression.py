'''
Implement linear regression with one variable on the California Housing dataset to predict
housing prices based on a single feature (e.g., the average number of rooms per dwelling).
Tasks:
● Load and preprocess the datase.
● Implement linear regression using both gradient descent and the normal equation.
● Evaluate the model performance using metrics such as Mean Squared Error
(MSE) and R-squared.
● Visualize the fitted line along with the data points.
'''

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

housing = fetch_california_housing()

X = housing.data
y = housing.target

for feature in range(X.shape[1]):

    print("=" * 50)
    print("Feature:", housing.feature_names[feature])
    print("=" * 100)

    x = X[:, feature]

    X_train, X_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42
    )

    mean = np.mean(X_train)
    std = np.std(X_train)

    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    train_mask = (X_train >= -3) & (X_train <= 3)
    test_mask = (X_test >= -3) & (X_test <= 3)

    X_train = X_train[train_mask]
    y_train = y_train[train_mask]

    X_test = X_test[test_mask]
    y_test = y_test[test_mask]

    theta0 = 0
    theta1 = 0

    learning_rate = 0.01
    iterations = 1000
    m = len(X_train)

    for i in range(iterations):
        y_pred = theta0 + theta1 * X_train
        error = y_train - y_pred
        theta0 = theta0 + (learning_rate / m) * np.sum(error)
        theta1 = theta1 + (learning_rate / m) * np.sum(error * X_train)

    y_test_pred = theta0 + theta1 * X_test

    mse = np.mean((y_test - y_test_pred) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_test - y_test_pred))
    r2 = r2_score(y_test, y_test_pred)

    print("Gradient Descent")
    print("Theta0:", theta0)
    print("Theta1:", theta1)
    print("MSE:", mse)
    print("RMSE:", rmse)
    print("MAE:", mae)
    print("R2:", r2)

    X_train_matrix = np.column_stack((np.ones(len(X_train)), X_train))

    theta = np.dot(
        np.dot(
            np.linalg.inv(
                np.dot(X_train_matrix.T, X_train_matrix)
            ),
            X_train_matrix.T
        ),
        y_train
    )

    X_test_matrix = np.column_stack((np.ones(len(X_test)), X_test))

    y_test_pred_normal = np.dot(X_test_matrix, theta)

    mse_normal = np.mean((y_test - y_test_pred_normal) ** 2)
    rmse_normal = np.sqrt(mse_normal)
    mae_normal = np.mean(np.abs(y_test - y_test_pred_normal))
    r2_normal = r2_score(y_test, y_test_pred_normal)

    print("Normal Equation")
    print("Theta0:", theta[0])
    print("Theta1:", theta[1])
    print("MSE:", mse_normal)
    print("RMSE:", rmse_normal)
    print("MAE:", mae_normal)
    print("R2:", r2_normal)

    sorted_index = np.argsort(X_test)

    X_sorted = X_test[sorted_index]
    y_sorted_gd = y_test_pred[sorted_index]
    y_sorted_ne = y_test_pred_normal[sorted_index]

    plt.figure(figsize=(8, 6))
    plt.scatter(X_test, y_test, color="green", s=8, alpha=0.5, label="Actual Data")
    plt.plot(X_sorted, y_sorted_gd, color="red", linewidth=2, label="Gradient Descent")
    plt.plot(X_sorted, y_sorted_ne, color="blue", linestyle="--", linewidth=2, label="Normal Equation")
    plt.xlabel(housing.feature_names[feature] + " (Normalized)")
    plt.ylabel("House Price")
    plt.title(housing.feature_names[feature])
    plt.legend()
    plt.grid(True)
    plt.show()