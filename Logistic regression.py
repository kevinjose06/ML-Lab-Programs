'''
Implement a logistic regression model to predict the likelihood of a disease using the Pima
Indians Diabetes dataset. Compare the performance with and without feature scaling.
Tasks:
● Load and preprocess the Pima Indians Diabetes dataset.
● Implement logistic regression for binary classification.
● Evaluate model performance with and without feature scaling.
● Analyze metrics such as accuracy, precision, recall, and F1-score.
'''

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

data = fetch_openml(name="diabetes", version=1, as_frame=False)
X = data.data.astype(float)
y = np.where(data.target == "tested_positive", 1, 0)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

model_without_scaling = LogisticRegression(max_iter=1000)
model_without_scaling.fit(X_train, y_train)
pred_without = model_without_scaling.predict(X_test)

acc_without = accuracy_score(y_test, pred_without)
prec_without = precision_score(y_test, pred_without)
rec_without = recall_score(y_test, pred_without)
f1_without = f1_score(y_test, pred_without)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model_with_scaling = LogisticRegression(max_iter=1000)
model_with_scaling.fit(X_train_scaled, y_train)
pred_with = model_with_scaling.predict(X_test_scaled)

acc_with = accuracy_score(y_test, pred_with)
prec_with = precision_score(y_test, pred_with)
rec_with = recall_score(y_test, pred_with)
f1_with = f1_score(y_test, pred_with)

print("\nPerformance Comparison")
print("\nWithout Feature Scaling")
print("Accuracy :", acc_without)
print("Precision:", prec_without)
print("Recall   :", rec_without)
print("F1-Score :", f1_without)

print("\nWith Feature Scaling")
print("Accuracy :", acc_with)
print("Precision:", prec_with)
print("Recall   :", rec_with)
print("F1-Score :", f1_with)

print("\nBest Performing Model\n")
if acc_without > acc_with:
    print("Without Feature Scaling performs better.")
elif acc_with > acc_without:
    print("With Feature Scaling performs better.")
else:
    print("Both models have the same accuracy.")