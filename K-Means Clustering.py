'''
Implement and apply K-means clustering to the Digits dataset. Experiment with different
numbers of clusters and evaluate the clustering results using metrics such as inertia and
silhouette score. Analyze how the choice of K affects clustering performance.
Tasks:
● Load and preprocess the Digits dataset.
● Implement K-means clustering with various numbers of clusters.
● Evaluate clustering performance using inertia and silhouette score.
● Analyze the impact of the number of clusters on clustering quality.
'''

import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

digits = load_digits()
X = StandardScaler().fit_transform(digits.data)

ks = range(2, 16)
inertia = []
silhouette = []

for k in ks:
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(X)

    inertia.append(model.inertia_)
    silhouette.append(silhouette_score(X, labels))

    print(k, "Inertia:", model.inertia_,
          "Silhouette:", silhouette[-1])

plt.plot(ks, inertia, marker='o')
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")
plt.title("Elbow Method")
plt.show()

plt.plot(ks, silhouette, marker='o')
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Score")
plt.show()