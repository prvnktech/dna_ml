import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

# Generate data for each class
economy_data = {
    'Horsepower': np.random.randint(70, 100, 17),
    'Weight': np.random.randint(1500, 2000, 17),
    'Engine Size': np.random.uniform(1.0, 2.0, 17),
    'MPG': np.random.uniform(25, 35, 17),
    'Make': ['Toyota'] * 17,
    'Category': 'Economy'
}

midsize_data = {
    'Horsepower': np.random.randint(150, 200, 17),
    'Weight': np.random.randint(2500, 3000, 17),
    'Engine Size': np.random.uniform(2.0, 3.0, 17),
    'MPG': np.random.uniform(20, 30, 17),
    'Make': ['Suzuki'] * 17,
    'Category': 'Midsize'
}

luxury_data = {
    'Horsepower': np.random.randint(250, 300, 16),
    'Weight': np.random.randint(3500, 4000, 16),
    'Engine Size': np.random.uniform(3.0, 4.0, 16),
    'MPG': np.random.uniform(15, 25, 16),
    'Make': ['BMW'] * 16,
    'Category': 'Luxury'
}

# Combine data into a single DataFrame
df_economy = pd.DataFrame(economy_data)
df_midsize = pd.DataFrame(midsize_data)
df_luxury = pd.DataFrame(luxury_data)

df = pd.concat([df_economy, df_midsize, df_luxury]).reset_index(drop=True)

# Encode 'Make' column
df['Make'] = df['Make'].astype('category').cat.codes

# Splitting the dataset into features and labels
X = df.drop('Category', axis=1)
y = df['Category']

# Encode the target labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Apply PCA to reduce to 2 components for visualization
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Initialize and train the Gradient Boosting Classifier
gbc = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
gbc.fit(X_train_pca, y_train)

# Predict the car categories on the test set
y_pred = gbc.predict(X_test_pca)

# Evaluate the model performance
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f'Accuracy: {accuracy:.2f}')
print('Classification Report:\n', report)
print('Confusion Matrix:\n', conf_matrix)

# Plot the first two principal components
plt.figure(figsize=(10, 7))
scatter = plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train, cmap='viridis', alpha=0.7)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA - Training Data')
plt.colorbar(scatter, ticks=[0, 1, 2], label='Category')
plt.show()
