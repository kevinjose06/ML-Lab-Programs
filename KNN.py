'''
Implement the K-Nearest Neighbors (KNN) algorithm for image classification using the
Fashion MNIST dataset. Experiment with different values of K and analyze their impact
on model performance.
Tasks:
● Load and preprocess the Fashion MNIST dataset.
● Implement KNN for multi-class classification.
● Experiment with different values of K and evaluate performance.
● Discuss the impact of different K values on model accuracy and computational
efficiency.
'''

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

fashion = fetch_openml("Fashion-MNIST",version=1,as_frame=False)

X = fashion.data
y = fashion.target.astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

k_values = [1,3, 5, 7]
results = []

for k in k_values:

    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test,y_pred)
    results.append(accuracy)

    print("K =", k)
    print("Accuracy :", accuracy)
    print()

best_index = results.index(max(results))
best_k = k_values[best_index]
best_accuracy = results[best_index]

print("\nBest K :", best_k)
print("Best Accuracy :", best_accuracy)