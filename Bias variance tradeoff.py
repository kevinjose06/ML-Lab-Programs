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
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

url = "https://lib.stat.cmu.edu/datasets/boston"
data = pd.read_csv(url, sep=r"\s+", skiprows=22, header=None)

X_all = np.hstack([data.values[::2, :], data.values[1::2, :2]])
y = data.values[1::2, 2]

X = X_all[:, [12]]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

degrees = range(1, 9)
train_errors = []
val_errors = []

for d in degrees:
    model = make_pipeline(StandardScaler(),PolynomialFeatures(degree=d),LinearRegression())
    model.fit(X_train, y_train)
    train_errors.append(mean_squared_error(y_train, model.predict(X_train)))
    val_errors.append(mean_squared_error(y_val, model.predict(X_val)))

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