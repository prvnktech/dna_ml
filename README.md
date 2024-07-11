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
