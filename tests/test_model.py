"""Unit tests for the Iris classification pipeline."""

from pathlib import Path

from iris_classifier.model import (
    evaluate,
    load_data,
    save_confusion_matrix,
    split_data,
    train_model,
)


def test_load_data_returns_150_flowers_with_4_features():
    X, y, labels = load_data()
    assert X.shape == (150, 4)
    assert len(y) == 150
    assert labels == ["setosa", "versicolor", "virginica"]


def test_split_data_keeps_20_percent_for_testing():
    X, y, _ = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    assert len(X_train) == 120
    assert len(X_test) == 30
    assert len(y_train) == 120
    assert len(y_test) == 30


def test_model_reaches_minimum_accuracy():
    X, y, _ = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    model = train_model(X_train, y_train)
    assert evaluate(model, X_test, y_test) >= 0.90


def test_save_confusion_matrix_creates_image(tmp_path: Path):
    X, y, labels = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    model = train_model(X_train, y_train)
    output = tmp_path / "cm.png"

    save_confusion_matrix(model, X_test, y_test, labels, path=output)

    assert output.exists()