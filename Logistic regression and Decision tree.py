'''
Implement and compare Logistic Regression and Decision Trees on the Adult Income 
dataset for predicting income levels. Evaluate both models based on performance metrics 
and interpretability. 
Tasks: 
● Load and preprocess the Adult Income dataset. 
● Implement both Logistic Regression and Decision Trees. 
● Compare the models based on metrics such as accuracy, precision, recall, and F1-
score. 
● Discuss the interpretability of both models and their suitability for the dataset.
'''

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

adult = fetch_openml("adult", version=2, as_frame=False)
X = adult.data
y = adult.target
y = np.where(np.char.strip(y.astype(str)) == ">50K", 1, 0)

numeric_features = [0, 2, 4, 10, 11, 12]
categorical_features = [1, 3, 5, 6, 7, 8, 9, 13]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

numeric_transformer = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
categorical_transformer = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", OneHotEncoder(handle_unknown="ignore"))])

preprocessor = ColumnTransformer([("num", numeric_transformer, numeric_features), ("cat", categorical_transformer, categorical_features)])

logistic = Pipeline([("preprocessor", preprocessor), ("model", LogisticRegression(max_iter=1000))])
logistic.fit(X_train, y_train)
logistic_pred = logistic.predict(X_test)

tree = Pipeline([("preprocessor", preprocessor), ("model", DecisionTreeClassifier(criterion="entropy", max_depth=5, random_state=42))])
tree.fit(X_train, y_train)
tree_pred = tree.predict(X_test)

def evaluate(name, y_true, y_pred):
    print("\n==========", name, "==========")
    print("Accuracy :", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred))
    print("Recall   :", recall_score(y_true, y_pred))
    print("F1 Score :", f1_score(y_true, y_pred))

evaluate("LOGISTIC REGRESSION", y_test, logistic_pred)
evaluate("DECISION TREE", y_test, tree_pred)

logistic_acc = accuracy_score(y_test, logistic_pred)
tree_acc = accuracy_score(y_test, tree_pred)
logistic_f1 = f1_score(y_test, logistic_pred)
tree_f1 = f1_score(y_test, tree_pred)

if logistic_acc > tree_acc:
    print("Logistic Regression has better Accuracy")
elif tree_acc > logistic_acc:
    print("Decision Tree has better Accuracy")
else:
    print("Both have the same Accuracy")

if logistic_f1 > tree_f1:
    print("Logistic Regression has better F1 Score")
elif tree_f1 > logistic_f1:
    print("Decision Tree has better F1 Score")
else:
    print("Both have the same F1 Score")
