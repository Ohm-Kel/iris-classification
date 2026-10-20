# Iris Classification

Classifies Iris flowers into three species (setosa, versicolor, virginica) from four measurements using logistic regression.

## How it works

1. **Load** the Iris dataset (150 flowers, 4 features: sepal length/width, petal length/width).
2. **Split** the data into 80% training and 20% test (`random_state=42` for reproducibility).
3. **Train** a logistic regression model on the training set.
4. **Evaluate** accuracy on the held-out test set.

## Results

| Metric   | Test set (30 samples) |
|----------|-----------------------|
| Accuracy | 1.00                  |

## Getting started

Prerequisites: Python 3.10+

```bash
git clone https://github.com/Ohm-Kel/iris-classification.git
cd iris-classification
python -m venv .venv
.venv\Scripts\Activate.ps1      # Windows
# source .venv/bin/activate     # macOS/Linux
pip install -r requirements.txt
python iris.py
```

## Limitations

The Iris dataset is small and easy to separate, so perfect test accuracy is expected and doesn't reflect how the model would perform on harder real-world data.

## Tech stack

Python, scikit-learn
