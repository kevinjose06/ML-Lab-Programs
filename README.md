# ML Lab Programs

This repository contains a collection of machine learning laboratory exercises implemented in Python. It is designed for semester-level study of the core ideas behind supervised learning, unsupervised learning, regularization, decision trees, support vector machines, clustering, and neural networks.

The programs here focus on the practical process of machine learning: preparing data, selecting a model, training, evaluating performance, comparing methods, and interpreting the results in a meaningful way.

## Course Context

These scripts align with the Machine Learning Lab curriculum and cover the key fundamentals expected in a modern ML laboratory:

- regression and prediction
- classification and discriminative modeling
- clustering and unsupervised structure discovery
- regularization and bias-variance tradeoff
- feature engineering and preprocessing
- neural network training and hyperparameter tuning
- comparison of multiple algorithms using standard evaluation metrics

## Learning Objectives

The goal of this repository is to help a learner understand:

1. How machine learning models make predictions from data.
2. How supervised and unsupervised methods differ in purpose and assumptions.
3. Why preprocessing steps like scaling, encoding, imputation, and train/test splitting are essential.
4. How accuracy, precision, recall, F1-score, MSE, MAE, and R² are used to judge model quality.
5. Why model choice, feature selection, regularization, and tuning strongly influence generalization performance.
6. How visualisation and statistical interpretation support decision-making in ML experiments.

---

## Repository Structure

The project includes the following core programs:

- [Simple linear regression.py](Simple%20linear%20regression.py)
- [Simple polynomial regression.py](Simple%20polynomial%20regression.py)
- [Ridge and Lasso regression.py](Ridge%20and%20Lasso%20regression.py)
- [Logistic regression.py](Logistic%20regression.py)
- [Logistic regression(MLE and MAP).py](Logistic%20regression%28MLE%20and%20MAP%29.py)
- [Logistic regression and Decision tree.py](Logistic%20regression%20and%20Decision%20tree.py)
- [KNN.py](KNN.py)
- [SVM.py](SVM.py)
- [SVM Classifier.py](SVM%20Classifier.py)
- [Naive-Bayes.py](Naive-Bayes.py)
- [ID3.py](ID3.py)
- [Agglomerative and K-Means.py](Agglomerative%20and%20K-Means.py)
- [Neural Network.py](Neural%20Network.py)
- [Multilayer Feed Forward Network.py](Multilayer%20Feed%20Forward%20Network.py)
- [Hyperparameter Tuning.py](Hyperparameter%20Tuning.py)

---

## Detailed Analysis of Each Experiment

### 1. Simple Linear Regression
File: [Simple linear regression.py](Simple%20linear%20regression.py)

This program demonstrates the simplest predictive model: fitting a line to data using the relationship

$y = mx + b$

The model learns the slope and intercept that minimize prediction error. The script uses a housing dataset, splits it into training and testing sets, fits a linear model, and evaluates performance using R² and residual/fit visualization.

Analysis:
- Useful for datasets with a nearly linear relationship between predictor and target.
- Helps visualize underfitting and how a single-variable model behaves.
- Provides the foundation for understanding regression, cost functions, and optimization.

### 2. Simple Polynomial Regression
File: [Simple polynomial regression.py](Simple%20polynomial%20regression.py)

This experiment extends linear regression by transforming input features into higher-degree terms such as $x^2$, $x^3$, etc. This allows the model to capture nonlinear relationships while still using linear regression in the transformed feature space.

Analysis:
- Demonstrates how nonlinear data can be modeled by feature expansion.
- Helps explain the tradeoff between model flexibility and overfitting.
- Shows why polynomial degree matters: too low leads to underfitting, too high leads to unstable behavior.

### 3. Ridge and Lasso Regression
File: [Ridge and Lasso regression.py](Ridge%20and%20Lasso%20regression.py)

This file compares two regularized regression models:

- Ridge regression adds an L2 penalty to prevent large coefficients.
- Lasso regression adds an L1 penalty, which encourages sparsity and can shrink some coefficients to zero.

Analysis:
- Useful when multicollinearity exists or when the model must avoid overfitting.
- Helps explain the bias-variance tradeoff.
- Shows how regularization improves generalization by constraining coefficient magnitudes.
- Lasso is especially useful for feature selection because it can reduce the number of active variables.

### 4. Logistic Regression
File: [Logistic regression.py](Logistic%20regression.py)

Logistic regression is a classification algorithm that models the probability of class membership using the sigmoid function.

Analysis:
- Appropriate for binary classification problems.
- Produces probabilities rather than direct class labels.
- Uses metrics such as accuracy, precision, recall, and F1-score.
- Suitable when the decision boundary is linear in feature space.

### 5. Logistic Regression with MLE and MAP
File: [Logistic regression(MLE and MAP).py](Logistic%20regression%28MLE%20and%20MAP%29.py)

This script implements logistic regression parameter estimation using:

- Maximum Likelihood Estimation (MLE)
- Maximum A Posteriori (MAP) estimation with regularization

The model compares the unregularized case with L1 and L2 regularized versions.

Analysis:
- MLE estimates parameters by maximizing likelihood from training data.
- MAP adds a prior assumption to parameters, preventing them from becoming excessively large.
- L1 regularization encourages sparse solutions.
- L2 regularization smooths coefficients and reduces variance.
- This is an excellent demonstration of statistical learning and the effect of priors on model behavior.

### 6. Logistic Regression and Decision Tree
File: [Logistic regression and Decision tree.py](Logistic%20regression%20and%20Decision%20tree.py)

This script compares two classification models:

- logistic regression
- decision tree classifier

The data undergoes preprocessing through pipelines, including imputation, scaling, and one-hot encoding for categorical values.

Analysis:
- Logistic regression performs well when class separation is linear and feature relationships are interpretable.
- Decision trees can model nonlinear decision boundaries and are easy to visualize.
- Pipeline preprocessing ensures transformations are applied consistently to train and test data.
- The comparison emphasizes the importance of model selection based on data type and complexity.

### 7. K-Nearest Neighbors (KNN)
File: [KNN.py](KNN.py)

KNN is a distance-based classification algorithm where a new sample is classified by majority vote among its nearest neighbors.

Analysis:
- Performance depends strongly on the choice of k.
- Works well for small-to-medium datasets but can become inefficient on large data.
- Scaling features is important because distance-based algorithms are sensitive to the magnitude of features.
- This program helps illustrate the concept of decision boundaries and local neighborhood structure.

### 8. Support Vector Machine (SVM)
Files: [SVM.py](SVM.py), [SVM Classifier.py](SVM%20Classifier.py)

These scripts study support vector machines, one of the most effective classifiers for high-dimensional data.

Analysis:
- SVM finds the hyperplane that maximizes the margin between classes.
- The kernel trick allows the method to model nonlinear separation.
- Scaling input features is essential for proper SVM performance.
- SVM is especially useful for classification problems with clear margin structure.
- The implementation demonstrates how margin-based learning differs from probabilistic or tree-based methods.

### 9. Naive Bayes
File: [Naive-Bayes.py](Naive-Bayes.py)

This experiment applies Naive Bayes to text classification using a document dataset.

Analysis:
- Naive Bayes works well for high-dimensional sparse data, especially text.
- It uses Bayes' theorem under the assumption that features are conditionally independent given the class.
- The script uses CountVectorizer and Multinomial/ Bernoulli variants to turn text into numerical features.
- It highlights why probabilistic classifiers are effective for sentiment and document categorization tasks.

### 10. ID3 Decision Tree
File: [ID3.py](ID3.py)

This program implements the ID3 algorithm using information gain as the splitting criterion.

Analysis:
- Decision trees recursively partition data based on the attribute that yields the greatest information gain.
- The algorithm favors attributes that reduce uncertainty the most.
- Easy to interpret and visualize because each split corresponds to a rule.
- Shows how tree depth and feature selection influence model complexity.

### 11. Agglomerative Clustering and K-Means
File: [Agglomerative and K-Means.py](Agglomerative%20and%20K-Means.py)

This file compares two clustering techniques:

- K-Means: centroid-based clustering that partitions data into k clusters.
- Agglomerative Clustering: hierarchical clustering that merges similar points step by step.

Analysis:
- Clustering is unsupervised learning; no class labels are provided.
- K-Means requires a predefined number of clusters and is sensitive to initialization.
- Agglomerative clustering builds a hierarchy and can reveal structure in a more interpretable way.
- Silhouette score is used to assess how well the clusters are separated.
- This experiment demonstrates the difference between partition-based and hierarchical clustering.

### 12. Neural Network
Files: [Neural Network.py](Neural%20Network.py), [Multilayer Feed Forward Network.py](Multilayer%20Feed%20Forward%20Network.py)

These scripts explore neural networks using MLP (Multi-Layer Perceptron) models.

Analysis:
- Neural networks learn nonlinear mappings through layers of weighted connections and activation functions.
- They are highly flexible and effective for complex patterns in image, speech, and structured data.
- Model performance depends on layer size, learning rate, number of epochs, and optimization behavior.
- The scripts illustrate how neural networks can be used for both classification and regression tasks.

### 13. Hyperparameter Tuning
File: [Hyperparameter Tuning.py](Hyperparameter%20Tuning.py)

This experiment focuses on selecting effective hyperparameters such as:

- learning rate
- batch size
- number of epochs
- hidden layer size

Analysis:
- Hyperparameters are not learned from data directly; they must be chosen by experimentation or search.
- Small changes in these values can significantly affect convergence speed and final accuracy.
- The script compares multiple configurations and measures training time and predictive accuracy.
- This demonstrates why tuning is a critical part of model development in real applications.

---

## Common Machine Learning Workflow Used Across the Project

Each script follows a similar pattern:

1. Load a dataset.
2. Inspect structure and target variable.
3. Preprocess the data (scaling, encoding, missing-value handling, feature transformation).
4. Split data into training and testing sets.
5. Train the model.
6. Predict on unseen data.
7. Evaluate using relevant metrics.
8. Interpret results and compare models.

This workflow is the foundation of practical machine learning and is repeated throughout the repository.

---

## Evaluation Metrics Used

The experiments rely on standard evaluation metrics such as:

- Accuracy: proportion of correctly classified samples.
- Precision: correctness of positive predictions.
- Recall: ability to find all relevant positive instances.
- F1-score: harmonic mean of precision and recall.
- Mean Squared Error (MSE): average squared prediction error.
- Mean Absolute Error (MAE): average absolute prediction error.
- R² Score: how much variance is explained by the model.
- Silhouette Score: clustering quality and separation measure.

---

## Common Libraries

The projects use the following Python libraries:

- numpy
- pandas
- matplotlib
- scikit-learn
- ucimlrepo

Depending on the dataset and experiment, additional packages may be used for data loading and plotting.

---

## Installation

If the dependencies are not already installed, run:

```bash
pip install numpy pandas matplotlib scikit-learn ucimlrepo
```

---

## How to Run a Program

From the project folder, run any script with Python:

```bash
python3 "Simple linear regression.py"
python3 "KNN.py"
python3 "SVM.py"
python3 "Hyperparameter Tuning.py"
```

Some scripts create plots and print model metrics in the console. Others may compare results across multiple models or configurations.

---

## Summary

This repository is a practical and educational collection of machine learning models that demonstrates how algorithms work, how they are trained, and how their performance is evaluated. It bridges theoretical concepts and implementation by showing real data experiments across regression, classification, clustering, regularization, and neural networks.

The value of the project lies not only in the final accuracy value, but in understanding why a model works, where it fails, and how modeling decisions affect the result.

---

## Notes

- These are educational implementations meant for learning and understanding algorithms, not production-optimized pipelines.
- Results may vary slightly depending on library versions and dataset availability.
- For research or deployment use, it is recommended to add cross-validation, model validation, and robust preprocessing pipelines.
