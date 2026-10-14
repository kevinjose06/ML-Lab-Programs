'''
Implement and train a Multilayer Feed-Forward Network (MLP) on the Wine Quality
dataset. Experiment with different numbers of hidden layers and neurons, and discuss how
these choices affect the network’s performance.
Tasks:
● Load and preprocess the Wine Quality dataset.
● Design and implement an MLP with varying architectures (different hidden layers
and neurons).
● Train and evaluate the network.
● Discuss the impact of architecture choices on performance.
'''

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score

wine = fetch_openml(name='wine-quality-red', version=1, as_frame=False)

X = wine.data
y = wine.target.astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

architectures = [(8,), (16,8), (24,16,8)]

for layers in architectures:
    model = MLPRegressor(hidden_layer_sizes=layers,max_iter=1000,random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("Architecture:", layers)
    print("MSE:", round(mse, 3))
    print("R2 Score:", round(r2, 3))
    print()