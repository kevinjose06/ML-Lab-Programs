# ML Lab Programs

This repository contains a collection of Python-based machine learning laboratory exercises covering the core concepts of supervised learning, unsupervised learning, ensemble methods, regularization, clustering, and neural networks. It is designed for academic learning and practical experimentation with widely used algorithms and evaluation techniques.

The programs in this project help learners understand how data is prepared, models are trained, performance is evaluated, and results are interpreted in real-world machine learning workflows.

## Overview

This collection includes implementations for:

- Regression and classification
- Clustering and dimensionality-aware learning
- Regularization techniques such as Ridge and Lasso
- Ensemble methods like Bagging and Boosting
- Decision tree algorithms and probabilistic models
- Support vector machines and k-nearest neighbors
- Neural networks and feed-forward architectures
- Hyperparameter tuning and model validation

## Repository Contents

### Core ML Programs

- [Agglomerative and K-Means.py](Agglomerative%20and%20K-Means.py) — comparison of agglomerative clustering and K-Means clustering
- [Bagging and Boosting.py](Bagging%20and%20Boosting.py) — ensemble learning using bagging and boosting techniques
- [Bias variance tradeoff.py](Bias%20variance%20tradeoff.py) — demonstration of bias-variance tradeoff and model complexity
- [Boostrapping and Cross Validation.py](Boostrapping%20and%20Cross%20Validation.py) — resampling strategies for estimating model performance
- [Hyperparameter Tuning.py](Hyperparameter%20Tuning.py) — tuning model parameters for better generalization
- [ID3.py](ID3.py) — decision tree implementation using ID3 and information gain
- [K-Means Clustering.py](K-Means%20Clustering.py) — centroid-based clustering with K-Means
- [KNN.py](KNN.py) — k-nearest neighbors classification
- [Logistic regression and Decision tree.py](Logistic%20regression%20and%20Decision%20tree.py) — comparison of logistic regression and decision tree classifiers
- [Logistic regression.py](Logistic%20regression.py) — binary classification using logistic regression
- [Logistic regression(MLE and MAP).py](Logistic%20regression%28MLE%20and%20MAP%29.py) — logistic regression with MLE and MAP estimation
- [Multilayer Feed Forward Network.py](Multilayer%20Feed%20Forward%20Network.py) — multilayer perceptron-style neural network implementation
- [Multinomial Distribution.py](Multinomial%20Distribution.py) — probabilistic modeling using multinomial distribution
- [Naive-Bayes.py](Naive-Bayes.py) — Naive Bayes classifier for probabilistic classification
- [Neural Network.py](Neural%20Network.py) — neural network model for learning nonlinear mappings
- [Ridge and Lasso regression.py](Ridge%20and%20Lasso%20regression.py) — regularized regression with L1 and L2 penalties
- [Simple linear regression.py](Simple%20linear%20regression.py) — basic linear regression model
- [Simple polynomial regression.py](Simple%20polynomial%20regression.py) — nonlinear regression using polynomial features
- [SVM Classifier.py](SVM%20Classifier.py) — support vector machine classification implementation
- [SVM.py](SVM.py) — support vector machine study and experiment

### Supporting Files

- [README.md](README.md) — project documentation and overview
- [test/simplelinreg.py](test/simplelinreg.py) — sample or validation script related to simple linear regression

## Typical Workflow Used in the Project

Each script follows a standard machine learning workflow:

1. Load a dataset
2. Explore and understand the data
3. Preprocess features and labels
4. Split data into training and testing sets
5. Train a model
6. Predict on unseen data
7. Evaluate performance using relevant metrics
8. Interpret the result and compare approaches

## Common Libraries

The programs use standard Python data-science libraries, including:

- numpy
- pandas
- matplotlib
- scikit-learn
- ucimlrepo

## Installation

Install the project dependencies with:

```bash
pip install numpy pandas matplotlib scikit-learn ucimlrepo
```

## Running the Programs

Run any script from the project directory:

```bash
python "Simple linear regression.py"
python "KNN.py"
python "SVM.py"
python "Hyperparameter Tuning.py"
```

Some scripts generate visual plots, while others print evaluation metrics or compare multiple models on the same dataset.

## Summary

This repository is a practical and educational collection of machine learning implementations that bridges theory and application. It helps students and learners understand algorithm behavior, model assumptions, data preprocessing, and evaluation methods across a broad range of ML topics.

The focus is not only on achieving high accuracy, but also on understanding why a model works, when it fails, and how design decisions affect performance.
