'''
Investigate the bias-variance tradeoff using polynomial regression on the Boston Housing
dataset. Plot the training and validation errors for various polynomial degrees and discuss
the tradeoff between bias and variance.
Tasks:
● Load and preprocess the Boston Housing dataset.
● Implement polynomial regression with varying degrees.
● Submission of Record: Students shall be allowed for the end semester examination only upon
submitting the duly certified record.
● Endorsement by External Examiner: The external examiner shall endorse the record
Course Outcomes (COs)
'''

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

data = fetch_openml(name='boston', version=1, as_frame=False)

X = data.data[:, [12]]
y = data.target.astype(float)

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.3, random_state=42
)

s = StandardScaler()
X_train = s.fit_transform(X_train)
X_val = s.transform(X_val)

degrees = range(1, 9)
train_errors = []
val_errors = []

for d in degrees:
    poly = PolynomialFeatures(degree=d)

    X_train_poly = poly.fit_transform(X_train)
    X_val_poly = poly.transform(X_val)

    model = LinearRegression()
    model.fit(X_train_poly, y_train)

    train_errors.append(mean_squared_error(y_train, model.predict(X_train_poly)))
    val_errors.append(mean_squared_error(y_val, model.predict(X_val_poly)))

    print("Degree:", d)
    print("Training Error:", train_errors[-1])
    print("Validation Error:", val_errors[-1])

plt.plot(degrees, train_errors, marker='o', label="Training Error")
plt.plot(degrees, val_errors, marker='s', label="Validation Error")

plt.xlabel("Polynomial Degree")
plt.ylabel("Mean Squared Error")
plt.title("Bias-Variance Tradeoff")
plt.legend()
plt.show()