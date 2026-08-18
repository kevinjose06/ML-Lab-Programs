'''
Implement a Naïve Bayes classifier to categorize text documents into topics using the 20
Newsgroups dataset. Compare the performance of Multinomial Naïve Bayes with
Bernoulli Naïve Bayes.
Tasks:
● Load and preprocess the 20 Newsgroups dataset.
● Implement Multinomial Naïve Bayes and Bernoulli Naïve Bayes classifiers.
● Evaluate and compare the performance of both models using metrics such as
accuracy and F1-score.
● Discuss the strengths and weaknesses of each Naïve Bayes variant for text
classification.
'''

from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.metrics import accuracy_score, f1_score

data = fetch_20newsgroups(subset='all')

X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

vectorizer = CountVectorizer(stop_words='english')

X_train_count = vectorizer.fit_transform(X_train)
X_test_count = vectorizer.transform(X_test)

multinomial_model = MultinomialNB()
multinomial_model.fit(X_train_count, y_train)
multinomial_pred = multinomial_model.predict(X_test_count)

multinomial_accuracy = accuracy_score(y_test, multinomial_pred)
multinomial_f1 = f1_score(y_test, multinomial_pred, average='macro')

binary_vectorizer = CountVectorizer(
    stop_words='english',
    binary=True
)
X_train_binary = binary_vectorizer.fit_transform(X_train)
X_test_binary = binary_vectorizer.transform(X_test)

bernoulli_model = BernoulliNB()
bernoulli_model.fit(X_train_binary, y_train)
bernoulli_pred = bernoulli_model.predict(X_test_binary)

bernoulli_accuracy = accuracy_score(y_test, bernoulli_pred)
bernoulli_f1 = f1_score(y_test, bernoulli_pred, average='macro')

print("\nResults")
print("Multinomial Naive Bayes")
print("Accuracy:", multinomial_accuracy)
print("F1-score:", multinomial_f1)

print("\nBernoulli Naive Bayes")
print("Accuracy:", bernoulli_accuracy)
print("F1-score:", bernoulli_f1)