'''
Implement and compare hierarchical (agglomerative) and partitional (K-means) clustering
algorithms on the Mall Customers dataset. Discuss the strengths and weaknesses of each
method based on clustering results and evaluation metrics.
Tasks:
● Load and preprocess the Mall Customers dataset.
● Apply both hierarchical (agglomerative) and K-means clustering.
● Compare results using metrics such as inertia, silhouette score, and clustering
visualization.
● Discuss the advantages and disadvantages of each clustering method.
'''

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score

url = "https://raw.githubusercontent.com/gakudo-ai/open-datasets/refs/heads/main/Mall_Customers.csv"
data = pd.read_csv(url)

X = data[['Annual Income (k$)', 'Spending Score (1-100)']]
scaler = StandardScaler()
X = scaler.fit_transform(X)

km = KMeans(n_clusters=5, random_state=42, n_init=10)
kmeans_labels = km.fit_predict(X)
hc = AgglomerativeClustering(n_clusters=5, linkage='ward')
hc_labels = hc.fit_predict(X)

print("K-means Inertia:", round(km.inertia_, 2))
print("K-means Silhouette Score:",round(silhouette_score(X, kmeans_labels), 3))
print("Agglomerative Silhouette Score:",round(silhouette_score(X, hc_labels), 3))

plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c=kmeans_labels)
plt.title("K-means Clustering")
plt.xlabel("Annual Income (scaled)")
plt.ylabel("Spending Score (scaled)")

plt.subplot(1, 2, 2)
plt.scatter(X[:, 0], X[:, 1], c=hc_labels)
plt.title("Agglomerative Clustering")
plt.xlabel("Annual Income (scaled)")
plt.ylabel("Spending Score (scaled)")

plt.tight_layout()
plt.show()