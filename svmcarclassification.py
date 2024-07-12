import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Generate synthetic data
np.random.seed(42)
num_samples = 100
num_items = 10

# Generating random feature vectors
X = np.random.randint(0, 2, (num_samples, num_items))

# Generating random labels: 1 for feminine, -1 for masculine, 0 for neutral
y = np.random.choice([1, -1, 0], num_samples)

# Adding 'Make' as a categorical feature (e.g., BMW, Suzuki, Toyota)
makes = ['BMW', 'Suzuki', 'Toyota']
make_column = np.random.choice(makes, num_samples)

# Creating a DataFrame
df = pd.DataFrame(X, columns=[f'Item_{i+1}' for i in range(num_items)])
df['Make'] = make_column
df['Label'] = y

# Data preprocessing
# Convert categorical 'Make' column to numerical using one-hot encoding
df = pd.get_dummies(df, columns=['Make'])

# Separate features and labels
X = df.drop('Label', axis=1).values
y = df['Label'].values

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train the SVM classifier
clf = svm.SVC(kernel='linear')
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("Classification Report:\n", report)

# Plotting the results (optional)
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Reduce dimensions to 2 for plotting
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Plotting the PCA-reduced data
plt.figure(figsize=(10, 6))
colors = {1: 'red', -1: 'blue', 0: 'green'}
for label in np.unique(y_train):
    plt.scatter(X_train_pca[y_train == label, 0], X_train_pca[y_train == label, 1],
                c=colors[label], label=f'Class {label}', alpha=0.5)

# Plotting the decision boundary
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()
xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 100), np.linspace(ylim[0], ylim[1], 100))
Z = clf.decision_function(pca.inverse_transform(np.c_[xx.ravel(), yy.ravel()]))
Z = Z.reshape(xx.shape)
ax.contourf(xx, yy, Z, levels=np.linspace(Z.min(), 0, 7), cmap=plt.cm.PuBu)
ax.contour(xx, yy, Z, levels=[0], linewidths=2, colors='darkred')

plt.title("SVM with Original Features (PCA used only for Plotting)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend()
plt.show()
