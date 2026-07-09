'''
Implement linear regression with one variable on the California Housing dataset to predict
housing prices based on a single feature (e.g., the average number of rooms per dwelling).
Tasks:
● Load and preprocess the datase.
● Implement linear regression using both gradient descent and the normal equation.
● Evaluate the model performance using metrics such as Mean Squared Error
(MSE) and R-squared.(MAE and RMS also added).
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

print("Shape of Features:", X.shape)
print("Shape of Target:", y.shape)

for i,value in enumerate(housing.feature_names):
    print(i,":",housing.feature_names[i])

# Use this commented section to find out the feature on which regression has to be applied.
"""
for i in range(X.shape[1]):
    plt.figure()
    plt.scatter(X[:,i],y,s=2)
    plt.xlabel(housing.feature_names[i])
    plt.ylabel("House price")
    plt.title(housing.feature_names[i])
    plt.show()
"""
X = X[:,0]
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
print("Training feature: ",X_train.shape)
print("test target: ",y_test.shape)


mean = np.mean(X_train)
std = np.std(X_train)
X_train = (X_train - mean) / std
X_test = (X_test - mean) / std
print(mean,std)

theta0 = 0
theta1 = 0
learning_rate = 0.01
iterations = 1000
m = len(X_train)

for i in range(iterations):
    y_pred = theta0 + theta1 * X_train
    error = y_train - y_pred
    theta0 = theta0 + (learning_rate/m) * np.sum(error)
    theta1 = theta1 + (learning_rate/m) * np.sum(error * X_train)

print("Theta0 =", theta0)
print("Theta1 =", theta1)

y_test_pred = theta0 + theta1 * X_test

mse = np.mean((y_test - y_test_pred) ** 2)
rmse = np.sqrt(mse)
mae = np.mean(np.abs(y_test - y_test_pred))
r2 = r2_score(y_test, y_test_pred)

print("Theta0 (Intercept):", theta0)
print("Theta1 (Slope):", theta1)

print("\nPerformance Metrics")

print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
print("Mean Absolute Error (MAE):", mae)
print("R-Squared Score (R²):", r2)

X_train_matrix = np.column_stack((np.ones(len(X_train)), X_train))
theta = np.linalg.inv(X_train_matrix.T @ X_train_matrix) @ X_train_matrix.T @ y_train
theta0_normal = theta[0]
theta1_normal = theta[1]
X_test_matrix = np.column_stack((np.ones(len(X_test)), X_test))
y_test_pred_normal = X_test_matrix @ theta

mse_normal = np.mean((y_test - y_test_pred_normal) ** 2)
rmse_normal = np.sqrt(mse_normal)
mae_normal = np.mean(np.abs(y_test - y_test_pred_normal))
r2_normal = r2_score(y_test, y_test_pred_normal)

print("\n========== Normal Equation Results ==========")

print("Theta0 (Intercept):", theta0_normal)
print("Theta1 (Slope):", theta1_normal)

print("\nPerformance Metrics")

print("Mean Squared Error (MSE):", mse_normal)
print("Root Mean Squared Error (RMSE):", rmse_normal)
print("Mean Absolute Error (MAE):", mae_normal)
print("R-Squared Score (R²):", r2_normal)

sorted_index = np.argsort(X_test)
X_sorted = X_test[sorted_index]

y_sorted_gd = y_test_pred[sorted_index]
y_sorted_ne = y_test_pred_normal[sorted_index]
plt.figure(figsize=(8,6))

plt.scatter(X_test,y_test,color="green",s=8,alpha=0.5,label="Actual Data")
plt.plot(X_sorted,y_sorted_gd,color="red",linewidth=2,label="Gradient Descent")
plt.plot(X_sorted,y_sorted_ne,color="blue",linestyle="--",linewidth=2,label="Normal Equation")

plt.xlabel("Median Income (Normalized)")
plt.ylabel("House Price")
plt.title("Simple Linear Regression")
plt.legend()
plt.grid(True)
plt.show()