"""
CodeAlpha - Task 1: Iris Flower Classification
Trains and evaluates a classifier that predicts Iris species from
sepal/petal measurements.

Usage:
    python iris_classification.py
    (optional) python iris_classification.py path/to/iris.csv

If no CSV path is given, the built-in scikit-learn Iris dataset is used,
which has identical columns to the CodeAlpha dataset.
"""

import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def load_data(csv_path=None):
    if csv_path:
        df = pd.read_csv(csv_path)
        df.columns = [c.strip() for c in df.columns]
        # drop non-feature columns like an Id/index column if present
        for id_col in ("Id", "id", "ID"):
            if id_col in df.columns:
                df = df.drop(columns=[id_col])
        # try to find the species/target column
        target_col = None
        for c in df.columns:
            if c.lower() in ("species", "class", "variety"):
                target_col = c
                break
        if target_col is None:
            raise ValueError("Could not find a species/class column in the CSV.")
        X = df.drop(columns=[target_col])
        y = df[target_col]
        if y.dtype == object:
            y = LabelEncoder().fit_transform(y)
        return X, y, list(X.columns)
    else:
        from sklearn.datasets import load_iris
        iris = load_iris()
        X = pd.DataFrame(iris.data, columns=iris.feature_names)
        y = iris.target
        return X, y, list(X.columns)


def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else None
    X, y, feature_names = load_data(csv_path)

    print("Dataset shape:", X.shape)
    print(X.describe())

    df_plot = X.copy()
    df_plot["species"] = y
    sns.pairplot(df_plot, hue="species")
    plt.savefig("iris_pairplot.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved iris_pairplot.png")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
        "Logistic Regression": LogisticRegression(max_iter=200),
    }

    best_name, best_acc, best_model = None, -1, None
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"\n=== {name} ===")
        print("Accuracy:", round(acc, 4))
        print(classification_report(y_test, y_pred))
        if acc > best_acc:
            best_name, best_acc, best_model = name, acc, model

    print(f"\nBest model: {best_name} (accuracy {best_acc:.4f})")

    y_pred_best = best_model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred_best)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix - {best_name}")
    plt.savefig("iris_confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved iris_confusion_matrix.png")


if __name__ == "__main__":
    main()
