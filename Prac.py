import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns


df = pd.read_csv('D:/Python and ML/CRWU bearing/test.csv')

X = df.drop('fault', axis=1)
y = df['fault']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(
        n_estimators=200,
        criterion='entropy',
        min_samples_split=5,
        max_depth=20,
        random_state=42
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print("accuracy = ", model.score(X_test, y_test))

#ploting results
plt.figure(figsize=(12, 10))
cm = confusion_matrix(y_test, y_pred)

# Get unique fault types in order
fault_types = y_test.unique()
fault_types_sorted = sorted(fault_types)  # Sort alphabetically

# Create heatmap
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=fault_types_sorted, 
            yticklabels=fault_types_sorted,
            square=True,
            cbar_kws={"shrink": 0.8})

plt.title('Confusion Matrix - Bearing Fault Classification', fontsize=16, fontweight='bold')
plt.xlabel('Predicted Fault Type', fontsize=12)
plt.ylabel('Actual Fault Type', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()
