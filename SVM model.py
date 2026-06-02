import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np

df = pd.read_csv('D:/Python and ML/CRWU bearing/test.csv')

# Separate features and labels
X = df.drop('fault', axis=1)
y = df['fault']

print("Features shape:", X.shape)
print("Feature names:", X.columns.tolist())
print("Label types:", y.unique())
print("\nFirst few labels:")
print(y.head())
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA to 2 dimensions
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print(f"\nVariance explained: PC1={pca.explained_variance_ratio_[0]:.1%}, PC2={pca.explained_variance_ratio_[1]:.1%}")


plt.figure(figsize=(12, 8))

fault_types = y.unique()
colors = plt.cm.tab10(range(len(fault_types)))

for fault, color in zip(fault_types, colors):
    mask = y == fault
    plt.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                label=fault, c=[color], alpha=0.7, s=30)

plt.xlabel(f'First Principal Component ({pca.explained_variance_ratio_[0]:.1%} variance)')
plt.ylabel(f'Second Principal Component ({pca.explained_variance_ratio_[1]:.1%} variance)')
plt.title('Data Visualization in 2D (PCA) - How Separable are Your Faults?')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Split data into training and testing sets
def train_svm(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = SVC(kernel='rbf', C=10, gamma=1, random_state=42)
    model.fit(X_train, y_train)
    return model, X_train, X_test, y_train, y_test

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    print("accuracy = ", model.score(X_test, y_test))


# Train on original 8D scaled data
svm_model, X_train, X_test, y_train, y_test = train_svm(X_scaled, y)

print("\n" + "="*60)
print("SVM PERFORMANCE ON ORIGINAL 8D DATA")
print("="*60)
evaluate_model(svm_model, X_test, y_test)

#ploting svm results
print("\n" + "="*60)
print("TRAINING SVM ON 2D PCA DATA FOR VISUALIZATION")
print("="*60)

# Split the PCA data
X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(X_pca, y, test_size=0.2, random_state=42)

# Train a NEW SVM model on the 2D data (this model expects 2 features)
svm_model_2d = SVC(kernel='rbf', C=10, gamma=1, random_state=42)
svm_model_2d.fit(X_train_pca, y_train_pca)

# Get accuracy on 2D data
accuracy_2d = svm_model_2d.score(X_test_pca, y_test_pca)
print(f"2D SVM Accuracy: {accuracy_2d:.1%}")
print(f"Note: This is lower than 8D accuracy ({svm_model.score(X_test, y_test):.1%}) because we lost information reducing to 2D")

print("\n" + "="*60)
print("PLOTTING SVM DECISION BOUNDARIES")
print("="*60)

# Create mesh grid for decision boundary
x_min, x_max = X_pca[:, 0].min() - 0.5, X_pca[:, 0].max() + 0.5
y_min, y_max = X_pca[:, 1].min() - 0.5, X_pca[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                     np.linspace(y_min, y_max, 300))

# IMPORTANT FIX: Use svm_model_2d (trained on 2D data) NOT svm_model (trained on 8D)
Z = svm_model_2d.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(14, 10))

# Get unique fault types and assign colors
fault_types = y.unique()
colors = plt.cm.tab10(range(len(fault_types)))
fault_to_color = {fault: colors[i] for i, fault in enumerate(fault_types)}
fault_to_int = {fault: i for i, fault in enumerate(fault_types)}

# Convert Z (string predictions) to integers for contourf
Z_int = np.vectorize(lambda x: fault_to_int.get(x, 0))(Z)

# Plot decision regions
contour = plt.contourf(xx, yy, Z_int, alpha=0.3, cmap='tab10', 
                       levels=range(len(fault_types)+1))

# Plot all data points
for fault in fault_types:
    mask = y == fault
    plt.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                label=fault, 
                c=[fault_to_color[fault]], 
                alpha=0.7, 
                s=40, 
                edgecolor='black', 
                linewidth=0.5)

# Labels and title
plt.xlabel(f'First Principal Component ({pca.explained_variance_ratio_[0]:.1%} variance)', 
           fontsize=12)
plt.ylabel(f'Second Principal Component ({pca.explained_variance_ratio_[1]:.1%} variance)', 
           fontsize=12)
plt.title(f'SVM Decision Boundaries on 2D PCA Data\nRBF Kernel, C=10, Gamma=1\n'
          f'Accuracy on 2D test set: {accuracy_2d:.1%} | Original 8D accuracy: {svm_model.score(X_test, y_test):.1%}', 
          fontsize=14, fontweight='bold')

# Legend
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
plt.grid(True, alpha=0.2)
plt.tight_layout()
plt.show()