'''
Implement and compare the performance of SVM classifiers with linear, polynomial, and
RBF kernels on the Fashion MNIST dataset. Analyze the advantages and disadvantages of
each kernel type.
Tasks:
● Load and preprocess the Fashion MNIST dataset.
● Implement SVM with linear, polynomial, and RBF kernels.
● Compare the classification performance for each kernel.
● Discuss the strengths and weaknesses of each kernel type.
'''

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

fashion_mnist = fetch_openml('Fashion-MNIST',version=1,as_frame=False)
X = fashion_mnist.data
y = fashion_mnist.target.astype(int)

X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=3000,test_size=500,stratify=y,random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

models = {"Linear": SVC(kernel='linear'),"Polynomial": SVC(kernel='poly', degree=3),"RBF": SVC(kernel='rbf')}

results = {}
for name, model in models.items():
    print("\nTraining", name, "SVM...")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    results[name] = accuracy

print("Linear SVM :", results["Linear"])
print("Polynomial SVM :", results["Polynomial"])
print("RBF SVM :", results["RBF"])

if results["Linear"] > results["Polynomial"] and results["Linear"] > results["RBF"]:
    print("Linear SVM has the highest accuracy.")
elif results["Polynomial"] > results["Linear"] and results["Polynomial"] > results["RBF"]:
    print("Polynomial SVM has the highest accuracy.")
elif results["RBF"] > results["Linear"] and results["RBF"] > results["Polynomial"]:
    print("RBF SVM has the highest accuracy.")
else:
    print("Two or more kernels have the same highest accuracy.")