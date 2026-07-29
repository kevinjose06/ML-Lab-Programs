'''
Estimate the parameters of a logistic regression model using MLE and MAP on the Breast
Cancer Wisconsin dataset. Compare the results and discuss the effects of regularization.
Tasks:
● Load and preprocess the dataset.
● Implement logistic regression with MLE.
● Apply MAP estimation with different regularization priors (L1 and L2
regularization).
● Compare the performance and parameter estimates with MLE and MAP.
'''

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

data = load_breast_cancer()

X = data.data
y = data.target
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def train_logistic_regression(X,y,regularization=None,lam=0,learning_rate=0.1,epochs=1000):
    m, n = X.shape

    theta = np.zeros(n)
    theta0 = 0

    for i in range(epochs):

        z = np.dot(X, theta) + theta0
        y_pred = sigmoid(z)
        error = y_pred - y

        dtheta = (1 / m) * np.dot(X.T, error)
        dtheta0 = (1 / m) * np.sum(error)

        if regularization == "L1":
            dtheta += (lam / n) * np.sign(theta)
        elif regularization == "L2":
            dtheta += (lam / n) * theta

        theta = theta - learning_rate * dtheta
        theta0 = theta0 - learning_rate * dtheta0

    return theta, theta0

def predict(X, theta, theta0):
    z = np.dot(X, theta) + theta0
    y_pred = sigmoid(z)
    prediction = (y_pred >= 0.5).astype(int)
    return prediction

def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

theta_mle, theta0_mle = train_logistic_regression(X_train,y_train)
theta_l1, theta0_l1 = train_logistic_regression(X_train,y_train,regularization="L1",lam=1)
theta_l2, theta0_l2 = train_logistic_regression(X_train,y_train,regularization="L2",lam=1)

pred_mle = predict(X_test, theta_mle, theta0_mle)
pred_l1 = predict(X_test, theta_l1, theta0_l1)
pred_l2 = predict(X_test, theta_l2, theta0_l2)

acc_mle = accuracy(y_test, pred_mle)
acc_l1 = accuracy(y_test, pred_l1)
acc_l2 = accuracy(y_test, pred_l2)

print("Accuracy Comparison\n")
print("MLE Accuracy      :", acc_mle)
print("L1 MAP Accuracy   :", acc_l1)
print("L2 MAP Accuracy   :", acc_l2)

print("Best performance:\n")
if acc_mle >= acc_l1 and acc_mle >= acc_l2:
    print("Model    : MLE")
elif acc_l1 >= acc_mle and acc_l1 >= acc_l2:
    print("Model    : MAP (L1)")
else:
    print("Model    : MAP (L2)")