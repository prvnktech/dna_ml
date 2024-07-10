# import the necessary packages
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Creating a fictional dataset of 200 cars with the make of the car, horsepower, engine capacity etc
np.random.seed(42)
makes = ['BMW', 'Suzuki', 'Toyota', 'Honda', 'Ford']
data = {
    'Make': np.random.choice(makes, 200),
    'Horsepower': np.random.randint(70, 300, 200),
    'Weight': np.random.randint(1500, 4000, 200),
    'Engine Size': np.random.uniform(1.0, 6.0, 200),
    'MPG': np.random.uniform(15, 35, 200),
    'Category': np.random.choice(['Economy', 'Midsize', 'Luxury'], 200)
}

df = pd.DataFrame(data)

# Splitting the dataset into features and labels
X = df.drop('Category', axis=1)
y = df['Category']

# Encode the target labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Encode the 'Make' column
X = pd.get_dummies(X, columns=['Make'], drop_first=True)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Apply PCA
pca = PCA(n_components=2)  # Choosing 2 components for visualization purposes
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
