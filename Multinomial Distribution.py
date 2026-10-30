'''
Use MLE and MAP to estimate the parameters of a multinomial distribution on the 20
Newsgroups dataset. Explore the impact of different priors on the estimation.
Tasks:
● Load and preprocess the dataset.
● Implement MLE for multinomial distribution parameter estimation.
● Apply MAP estimation with various priors (e.g., Dirichlet priors).
● Compare results and evaluate the effect of different priors.
'''

import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer

data = fetch_20newsgroups(subset='train',categories=['sci.space', 'rec.autos'],remove=('headers', 'footers', 'quotes'))

vectorizer = CountVectorizer(max_features=1000, stop_words='english')
X = vectorizer.fit_transform(data.data)

counts = np.asarray(X.sum(axis=0)).flatten()
words = vectorizer.get_feature_names_out()
N = counts.sum()
V = len(counts)

mle = counts / N
for alpha in [1, 2, 10]:
    map_est = (counts + alpha - 1) / (N + V * (alpha - 1))

    print("\nAlpha =", alpha)
    for i in np.argsort(map_est)[-5:][::-1]:
        print(words[i], "MLE:", round(mle[i], 5),
              "MAP:", round(map_est[i], 5))