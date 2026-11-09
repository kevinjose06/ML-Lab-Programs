'''
Implement bagging and boosting ensemble methods on the Titanic dataset. Compare the
performance of both methods in terms of accuracy, precision, recall, and F1-score.
Discuss how each method improves model performance and their respective strengths and
weaknesses.
Tasks:
● Load and preprocess the Titanic dataset.
● Implement bagging using a base classifier (e.g., decision tree) and evaluate
performance.
● Implement boosting using a boosting algorithm (e.g., AdaBoost) and evaluate
performance.
● Compare performance metrics and discuss the strengths and weaknesses of each
method.
'''

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier,AdaBoostClassifier
from sklearn.metrics import accuracy_score

t = fetch_openml(name = "titanic",version = 1,as_frame = True)
x = t.data
y = t.target.astype(int)

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 42)

nf = [3,4,5,7]
cf = [0,2,9]

nt = Pipeline([("imputer",SimpleImputer(strategy='median')),("scaler",StandardScaler())])
ct = Pipeline([("imputer",SimpleImputer(strategy='most_frequent')),("encoder",OneHotEncoder(handle_unknown="ignore"))])

p = ColumnTransformer([("num",nt,nf),("cat",ct,cf)])

bc = Pipeline([("preprocessor",p),("model",BaggingClassifier(estimator=DecisionTreeClassifier(max_depth=2,criterion="entropy",random_state=42),n_estimators = 5,random_state=42))])
bc.fit(x_train,y_train)
bcp = bc.predict(x_test)
print("Bagging accuracy: ",accuracy_score(y_test,bcp))

ac = Pipeline([("preprocessor",p),("model",AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=2,criterion="entropy",random_state=42),n_estimators = 5,random_state = 42))])
ac.fit(x_train,y_train)
acp = ac.predict(x_test)
print("Boosting accuracy: ",accuracy_score(y_test,acp))