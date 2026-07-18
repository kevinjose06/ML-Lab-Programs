'''
Implement polynomial regression on the Auto MPG dataset to predict miles per gallon
(MPG) based on engine displacement. Compare polynomial regression results with linear
regression.
Tasks:
● Load and preprocess the dataset.
● Implement polynomial regression of varying degrees.
● Compare the polynomial regression models with linear regression using metrics
such as MSE and R-squared.
● Visualize the polynomial fit.
'''

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

auto = fetch_openml(name="autoMpg",version=1,as_frame=False)
x = auto.data[:,1].astype(float)
y = auto.target.astype(float)
print(np.isnan(x).sum())
print(np.isnan(y).sum())

'''
If there are any missing values use:
mask = ~np.isnan(x)
x = x[mask]
y = y[mask]
'''

x_train,x_test,y_train,y_test = train_test_split(
    x,y,test_size=0.2,random_state=42
)

mean = np.mean(x_train)
std = np.std(x_train)
x_train = (x_train - mean) / std
x_test = (x_test - mean) / std
x_train = x_train.reshape(-1,1)
x_test = x_test.reshape(-1,1)

linear_model = LinearRegression()
linear_model.fit(x_train,y_train)
y_pred_linear = linear_model.predict(x_test)

mse = np.mean((y_test - y_pred_linear) ** 2)
rmse = np.sqrt(mse)
mae = np.mean(np.abs(y_test - y_pred_linear))
r2 = r2_score(y_test,y_pred_linear)
print("Linear Regression")
print("MSE:", mse)
print("RMSE:", rmse)
print("MAE:", mae)
print("R2:", r2)

print("Polynomial Regression")
for i in [2, 3, 4, 5]:

    poly = PolynomialFeatures(degree=i)

    x_train_poly = poly.fit_transform(x_train)
    x_test_poly = poly.transform(x_test)

    poly_model = LinearRegression()
    poly_model.fit(x_train_poly, y_train)
    y_pred_poly = poly_model.predict(x_test_poly)

    mse_poly = np.mean((y_test - y_pred_poly) ** 2)
    rmse_poly = np.sqrt(mse_poly)
    mae_poly = np.mean(np.abs(y_test - y_pred_poly))
    r2_poly = r2_score(y_test,y_pred_poly)

    print("=" * 50)
    print("Polynomial Regression (Degree =", i, ")")
    print("=" * 50)
    print("MSE:", mse_poly)
    print("RMSE:", rmse_poly)
    print("MAE:", mae_poly)
    print("R2:", r2_poly)

    # Print results