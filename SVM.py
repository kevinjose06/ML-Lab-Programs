'''
Implement a Linear Support Vector Machine (SVM) to classify the Iris dataset. Visualize
the decision boundary and discuss how the margin is determined.
Tasks:
● Load and preprocess the Iris dataset.
● Implement a Linear SVM for binary classification (e.g., classify Setosa vs. Non-
Setosa).
● Visualize the decision boundary and margin.
● Discuss the concept of the margin and how it influences classification.
'''
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

iris = load_iris()
x = iris.data[:,:2]
y = (iris.target!=0).astype(int)

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42, test_size=0.2, stratify=y)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

model = SVC(kernel='linear', C=1.0)
model.fit(x_train, y_train)
pred = model.predict(x_test)

print("Accuracy Score:",accuracy_score(y_test, pred),"\n");

w=model.coef_[0]
b = model.intercept_[0]

x_min = x_train[:,0].min() - 1
x_max = x_train[:,0].max() + 1
x = np.linspace(x_min, x_max, 100)

y_bound = -(w[0]*x + b)/w[1]
y_margin1 = -(w[0]*x + b -1)/w[1]
y_margin2 = -(w[0]*x + b +1)/w[1]

plt.scatter(x_train[y_train==0,0], x_train[y_train==0,1], label='Setosa')
plt.scatter(x_train[y_train==1, 0], x_train[y_train==1, 1], label = 'Non-Setosa')
plt.plot(x, y_bound, label='Decision Boundary')
plt.plot(x, y_margin1, '--', label='Margin 1')
plt.plot(x, y_margin2, '--', label='Margin 2')

plt.scatter(model.support_vectors_[:,0], model.support_vectors_[:,1], s=100, facecolors='none', edgecolors='black', label="Support Vectors")

plt.xlabel('Standardized Sepal Length')
plt.ylabel("Standardized Sepal Width")
plt.title("Linear SVM: Setosa vs Non-Setosa")
plt.legend()
plt.show()