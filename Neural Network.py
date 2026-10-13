'''
Implement and compare the performance of a neural network using different activation
functions (Sigmoid, ReLU, Tanh) on the MNIST dataset. Analyze how each activation
function affects the training process and classification accuracy.
Tasks:
● Load and preprocess the MNIST dataset.
● Implement neural networks using Sigmoid, ReLU, and Tanh activation functions.
● Train and evaluate each network.
● Compare training times, convergence, and classification accuracy.
'''

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import time

mnist = fetch_openml(name='mnist_784', version=1, as_frame=False)
X = mnist.data[:10000].astype('float32') / 255
y = mnist.target[:10000].astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

activations = ['logistic', 'relu', 'tanh']

for act in activations:
    model = MLPClassifier(hidden_layer_sizes=(64,),activation=act,max_iter=20,random_state=42)
    start = time.time()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    end = time.time()

    accuracy = accuracy_score(y_test, y_pred)

    print("Activation:", act)
    print("Accuracy:", round(accuracy * 100, 2), "%")
    print("Training time:", round(end - start, 2), "seconds")
    print("Iterations:", model.n_iter_)
    print()
