"""Train and evaluate a logistic regression classifier on the Iris dataset."""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

RANDOM_STATE = 42
TEST_SIZE = 0.2
CONFUSION_MATRIX_PATH = Path("docs/confusion_matrix.png")


def load_data() -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return the Iris features, labels and species names."""
    iris = load_iris()
    return iris.data, iris.target, list(iris.target_names)


def split_data(
    X: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split features and labels into reproducible train and test sets."""
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)


def train_model(X_train: np.ndarray, y_train: np.ndarray) -> LogisticRegression:
    """Fit a logistic regression model on the training data."""
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)
    return model


def evaluate(model: LogisticRegression, X_test: np.ndarray, y_test: np.ndarray) -> float:
    """Return the model's accuracy on the test set."""
    return accuracy_score(y_test, model.predict(X_test))


def save_confusion_matrix(
    model: LogisticRegression,
    X_test: np.ndarray,
    y_test: np.ndarray,
    labels: list[str],
    path: Path = CONFUSION_MATRIX_PATH,
) -> None:
    """Plot the test-set confusion matrix and save it as an image."""
    path.parent.mkdir(parents=True, exist_ok=True)
    ConfusionMatrixDisplay.from_estimator(
        model, X_test, y_test, display_labels=labels, cmap="Greys"
    )
    plt.title("Confusion matrix (test set)")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main() -> None:
    """Run the full pipeline: load, split, train, evaluate, plot."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    X, y, labels = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    model = train_model(X_train, y_train)

    accuracy = evaluate(model, X_test, y_test)
    logger.info("Model accuracy: %.2f", accuracy)

    save_confusion_matrix(model, X_test, y_test, labels)
    logger.info("Saved confusion matrix to %s", CONFUSION_MATRIX_PATH)


if __name__ == "__main__":
    main()