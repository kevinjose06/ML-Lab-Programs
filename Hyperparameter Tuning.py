'''
Implement and perform hyperparameter tuning for a neural network on the Fashion
MNIST dataset. Experiment with different learning rates, batch sizes, and epochs, and
discuss the impact on model performance.
Tasks:
● Load and preprocess the Fashion MNIST dataset.
● Experiment with different hyperparameters (learning rate, batch size, epochs).
● Train and evaluate the network.
● Discuss how hyperparameter choices affect model performance.
'''

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import time

data = fetch_openml(name='Fashion-MNIST', version=1, as_frame=False)
X = data.data[:10000].astype('float32') / 255
y = data.target[:10000].astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

settings = [(0.001, 64, 10),(0.01, 64, 10),(0.001, 128, 20)]

for lr, batch, epochs in settings:
    model = MLPClassifier(hidden_layer_sizes=(64,),learning_rate_init=lr,batch_size=batch,max_iter=epochs,random_state=42)
    start = time.time()
    model.fit(X_train, y_train)
    end = time.time()
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    print("Learning rate:", lr)
    print("Batch size:", batch)
    print("Epochs:", model.n_iter_)
    print("Accuracy:", round(accuracy * 100, 2), "%")
    print("Training time:", round(end - start, 2), "seconds")
    print()