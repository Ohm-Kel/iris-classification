from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score
from sklearn.model_selection import train_test_split

# 1. Load: 150 flowers, 4 measurements each (X), species label (y)
iris = load_iris()
X, y = iris.data, iris.target

# 2. Split: 80% to train on, 20% held back to test on unseen data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Train: the model learns the link between measurements and species
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# 4. Evaluate: predict the test flowers and compare with the true labels
y_pred = model.predict(X_test)
print(f"Model accuracy: {accuracy_score(y_test, y_pred):.2f}")

# 5. Confusion matrix: rows = true species, columns = predicted species
Path("docs").mkdir(exist_ok=True)
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred, display_labels=iris.target_names, cmap="Greys"
)
plt.title("Confusion matrix (test set)")
plt.tight_layout()
plt.savefig("docs/confusion_matrix.png", dpi=150)
print("Saved confusion matrix to docs/confusion_matrix.png")