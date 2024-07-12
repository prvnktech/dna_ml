# dna_ml
Machine Learning projects for Digital Nexus AI

Gradient Boosting Classifier

**Problem Statement:** 
Classify cars into different categories (e.g., "Economy", "Midsize", "Luxury") based on various features such as horsepower, weight, engine size, etc.

**Explanation**
- Data Generation: We generated synthetic data for three distinct car categories (Economy, Midsize, Luxury) with the Make column.
- Preprocessing: We encoded the Make column, encoded the target labels, split the data into training and testing sets, and standardized the features.
- PCA: We applied PCA to reduce the dataset to 2 principal components for visualization.
- Model Training: We trained a GradientBoostingClassifier on the transformed training data.
- Evaluation: We evaluated the model using accuracy, classification report, and confusion matrix.
- Visualization: We plotted the first two principal components to visualize the data.

**Data Generation:**
Generates synthetic data for 100 individuals picking 10 items, with labels representing feminine, masculine, and neutral classes.
Adds a 'Make' column representing car brands as a categorical feature.
Data Preprocessing:

Uses one-hot encoding to convert the categorical 'Make' column to numerical form.
Standardizes the features for better performance of the SVM.
Data Splitting:

Splits the data into training and test sets.
Model Training:

Trains an SVM classifier using the training data.
Model Evaluation:

Evaluates the classifier on the test data and prints accuracy and classification report.
Visualization:

Optionally plots the data using PCA only for visualization purposes (not for model training).
This program provides a complete pipeline for classification using SVM, including preprocessing, training, evaluation, and visualization of the results.
