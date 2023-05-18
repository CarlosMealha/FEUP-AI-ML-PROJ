# Sports Articles Objectivity Analysis

This project focuses on applying supervised learning algorithms to analyze the objectivity of sports articles. The goal is to classify sports articles as either objective or subjective based on their content. The project utilizes the following algorithms: K-Nearest Neighbors (KNN), Decision Trees, and Support Vector Classifier (SVC).

## Dataset

The dataset used in this project consists of a collection of sports articles. Each article is labeled as either objective or subjective. The dataset includes various features extracted from the articles, such as word frequency, sentence length, and sentiment scores. These features serve as inputs for training and evaluating the supervised learning models.

## Algorithms

The project implements the following supervised learning algorithms:

1. K-Nearest Neighbors (KNN): KNN is a non-parametric algorithm that classifies a new data point based on the majority class of its k nearest neighbors in the feature space. It measures the distance between data points to determine their similarity and makes predictions based on the most prevalent class among the nearest neighbors.

2. Decision Trees: Decision Trees are a versatile algorithm that builds a tree-like model for classification. It partitions the feature space based on the values of different features and creates decision rules to assign class labels. The model learns to make decisions by asking questions at each node of the tree until reaching a leaf node with a predicted class label.

3. Support Vector Classifier (SVC): SVC is a supervised learning algorithm that separates data points into different classes by finding an optimal hyperplane in a high-dimensional feature space. In this project, SVC is used to classify sports articles by creating a decision boundary that maximally separates the objective and subjective articles based on their textual features.

## Implementation

The project implementation can be found in the "deep-learning-texts.ipynb" notebook. The notebook covers the following steps:

1. Data preprocessing: Loading the dataset, performing feature extraction, and splitting the data into training and test sets.

2. Algorithm selection: Defining the KNN, Decision Trees, and SVC.

3. Model training: Fitting the algorithms to the training data to learn the underlying patterns and relationships.

4. Model evaluation: Evaluating the trained models on the test data using accuracy as the performance metric. The accuracy score measures the proportion of correctly classified articles.

5. Visual comparison: Logging the accuracy and other relevant metrics for each algorithm to facilitate a visual comparison of their performance.

## Results

The results of the supervised learning models applied to the sports articles dataset are as follows:

- K-Nearest Neighbors achieved an accuracy of 68.4% on the test set.
- Decision Trees achieved an accuracy of 72.8% on the test set.
- Support Vector Machines achieved an accuracy of 77.6% on the test set.


## Requirements

The following dependencies are required to run the project:

- scikit-learn
- pandas
- numpy
- matplotlib

