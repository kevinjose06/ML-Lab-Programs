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
from sklearn.linear_model import LinearRegression,Ridge,RidgeCV,Lasso,LassoCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

linear = LinearRegression()
linear.fit(X_train, y_train)
linear_pred = linear.predict(X_test)

ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)
ridge_pred = ridge.predict(X_test)

ridge_cv = RidgeCV(alphas=[0.01, 0.1, 1, 10, 100], cv=5)
ridge_cv.fit(X_train, y_train)
ridgecv_pred = ridge_cv.predict(X_test)

lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)
lasso_pred = lasso.predict(X_test)

lasso_cv = LassoCV(cv=5, random_state=42)
lasso_cv.fit(X_train, y_train)
lassocv_pred = lasso_cv.predict(X_test)

def evaluate(name, y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print(name,":")
    print("MSE  :", mse)
    print("RMSE :", rmse)
    print("MAE  :", mae)
    print("R²   :", r2)

evaluate("Linear Regression", y_test, linear_pred)
print()
evaluate("Ridge Regression", y_test, ridge_pred)
print("Alpha :", ridge.alpha)
print()
evaluate("Ridge Regression (CV)", y_test, ridgecv_pred)
print("Best Alpha :", ridge_cv.alpha_)
print()
evaluate("Lasso Regression", y_test, lasso_pred)
print("Alpha :", lasso.alpha)
print()
evaluate("Lasso Regression (CV)", y_test, lassocv_pred)
print("Best Alpha :", lasso_cv.alpha_)