# ML Lab Programs

This repository contains a collection of Python-based machine learning exercises designed for academic study, practice, and experimentation. The programs cover several core areas of machine learning, including regression, classification, clustering, ensemble methods, regularization, and neural networks.

The goal of this project is to help learners understand how models are built, trained, evaluated, and interpreted in real-world machine learning workflows.

## Overview

The repository includes implementations for:

- Regression and classification techniques
- Clustering methods such as K-Means and agglomerative clustering
- Model evaluation strategies including bootstrapping and cross-validation
- Regularization methods such as Ridge and Lasso
- Ensemble learning using bagging and boosting
- Decision trees and probabilistic models
- Support vector machines and k-nearest neighbors
- Neural networks and multilayer feed-forward learning
- Hyperparameter tuning and model comparison

## Repository Contents

### Core ML Programs

- [Agglomerative and K-Means.py](Agglomerative%20and%20K-Means.py) — comparison of agglomerative clustering and K-Means clustering
- [Bagging and Boosting.py](Bagging%20and%20Boosting.py) — ensemble learning using bagging and boosting techniques
- [Bias variance tradeoff.py](Bias%20variance%20tradeoff.py) — study of bias-variance tradeoff and model complexity
- [Boostrapping and Cross Validation.py](Boostrapping%20and%20Cross%20Validation.py) — resampling methods for model evaluation
- [Hyperparameter Tuning.py](Hyperparameter%20Tuning.py) — parameter optimization for better model performance
- [ID3.py](ID3.py) — decision tree implementation based on ID3 and information gain
- [K-Means Clustering.py](K-Means%20Clustering.py) — centroid-based clustering using K-Means
- [KNN.py](KNN.py) — k-nearest neighbors classification
- [Logistic regression and Decision tree.py](Logistic%20regression%20and%20Decision%20tree.py) — comparison of logistic regression and decision tree classifiers
- [Logistic regression.py](Logistic%20regression.py) — binary classification using logistic regression
- [Logistic regression(MLE and MAP).py](Logistic%20regression%28MLE%20and%20MAP%29.py) — logistic regression using MLE and MAP estimation
- [Multilayer Feed Forward Network.py](Multilayer%20Feed%20Forward%20Network.py) — multilayer perceptron-style neural network implementation
- [Multinomial Distribution.py](Multinomial%20Distribution.py) — probabilistic modeling using the multinomial distribution
- [Naive-Bayes.py](Naive-Bayes.py) — Naive Bayes classifier for probabilistic prediction
- [Neural Network.py](Neural%20Network.py) — neural network model for learning nonlinear relationships
- [Ridge and Lasso regression.py](Ridge%20and%20Lasso%20regression.py) — regularized linear regression with L1 and L2 penalties
- [Simple linear regression.py](Simple%20linear%20regression.py) — basic linear regression example
- [Simple polynomial regression.py](Simple%20polynomial%20regression.py) — polynomial regression for nonlinear data patterns
- [SVM Classifier.py](SVM%20Classifier.py) — support vector machine classification model
- [SVM.py](SVM.py) — support vector machine experiment and analysis

### Supporting Files

- [README.md](README.md) — project documentation and overview
- [.gitignore](.gitignore) — repository ignore configuration

## Typical ML Workflow Used in the Project

Each program follows a standard machine learning workflow:

1. Load and inspect the dataset
2. Prepare and transform the data
3. Split data into training and testing sets
4. Train the model
5. Generate predictions on new data
6. Evaluate the model using suitable metrics
7. Interpret the results and compare approaches

## Common Libraries

These programs typically use the following Python libraries:

- numpy
- pandas
- matplotlib
- scikit-learn
- ucimlrepo

## Installation

Install the required dependencies with:

```bash
pip install numpy pandas matplotlib scikit-learn ucimlrepo
```

## Running the Programs

Run a script from the project directory using Python:

```bash
python "Simple linear regression.py"
python "KNN.py"
python "SVM.py"
python "Hyperparameter Tuning.py"
```

Some scripts produce visual plots, while others print evaluation metrics or compare multiple models on the same dataset.

## Summary

This repository is a practical collection of machine learning implementations that connects theory with hands-on application. It is useful for students, learners, and anyone interested in understanding how core algorithms behave under different data and modeling conditions.

The emphasis is not only on accuracy, but also on understanding model assumptions, data preprocessing, and the effect of design choices on predictive performance.
