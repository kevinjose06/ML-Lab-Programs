'''
Implement bootstrapping and cross-validation on the Iris dataset. Compare the model
performance metrics (e.g., accuracy, F1-score) obtained using these resampling methods.
Discuss the advantages and disadvantages of each method.
Tasks:
● Load and preprocess the Iris dataset.
● Implement bootstrapping to generate multiple samples and evaluate the model.
● Implement k-fold cross-validation and evaluate the model.
● Compare the performance metrics and discuss the pros and cons of each
resampling method.
'''

import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_validate

iris = load_iris()
X, y = iris.data, iris.target

model = LogisticRegression(max_iter=500)
rng = np.random.default_rng(42)

acc = []
f1 = []

for i in range(100):
    train = rng.choice(len(X), len(X), replace=True)
    test = np.setdiff1d(np.arange(len(X)), np.unique(train))

    model.fit(X[train], y[train])
    pred = model.predict(X[test])

    acc.append(accuracy_score(y[test], pred))
    f1.append(f1_score(y[test], pred, average='macro'))

print("Bootstrap Accuracy:", np.mean(acc))
print("Bootstrap F1-score:", np.mean(f1))

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_validate(model, X, y, cv=cv, scoring=['accuracy', 'f1_macro'])

print("CV Accuracy:", scores['test_accuracy'].mean())
print("CV F1-score:", scores['test_f1_macro'].mean())