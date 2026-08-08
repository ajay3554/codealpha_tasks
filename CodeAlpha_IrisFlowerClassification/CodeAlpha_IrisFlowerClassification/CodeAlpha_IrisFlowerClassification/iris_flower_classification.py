"""
CodeAlpha Data Science Internship - Task 1
Iris Flower Classification
--------------------------------------------------
Trains a classifier to identify Iris species (setosa, versicolor,
virginica) using sepal/petal measurements.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

RANDOM_STATE = 42


def load_data():
    """Load the real Iris.csv (Kaggle-format) if present, else fall back
    to the built-in scikit-learn Iris dataset."""
    csv_path = "Iris.csv"
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        print(f"Loaded real dataset: {csv_path}")
        if "Id" in df.columns:
            df = df.drop(columns=["Id"])
        rename_map = {
            "SepalLengthCm": "sepal length (cm)",
            "SepalWidthCm": "sepal width (cm)",
            "PetalLengthCm": "petal length (cm)",
            "PetalWidthCm": "petal width (cm)",
            "Species": "species",
        }
        df = df.rename(columns=rename_map)
        df["species"] = df["species"].str.replace("Iris-", "", regex=False)
        species_to_code = {name: code for code, name in enumerate(sorted(df["species"].unique()))}
        df["target"] = df["species"].map(species_to_code)
        return df

    print(f"'{csv_path}' not found - using scikit-learn's built-in Iris dataset.")
    iris = load_iris(as_frame=True)
    df = iris.frame.copy()
    df["species"] = df["target"].map(dict(enumerate(iris.target_names)))
    return df


def explore_data(df):
    print("\n--- Dataset shape ---")
    print(df.shape)
    print("\n--- First 5 rows ---")
    print(df.head())
    print("\n--- Summary statistics ---")
    print(df.describe())
    print("\n--- Class distribution ---")
    print(df["species"].value_counts())

    # Pairplot to visualize feature separability
    sns.pairplot(df.drop(columns=["target"]), hue="species")
    plt.savefig("iris_pairplot.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Correlation heatmap
    plt.figure(figsize=(6, 5))
    sns.heatmap(df.drop(columns=["target", "species"]).corr(), annot=True, cmap="Blues")
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig("iris_correlation_heatmap.png", dpi=150)
    plt.close()

    print("\nSaved plots: iris_pairplot.png, iris_correlation_heatmap.png")


def train_and_evaluate(df):
    feature_cols = [c for c in df.columns if c not in ("target", "species")]
    X = df[feature_cols]
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    # Scale features (good practice even though tree models don't strictly need it)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)

    acc = accuracy_score(y_test, y_pred)
    print(f"\n--- Model: RandomForestClassifier ---")
    print(f"Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=df["species"].unique()))

    # Confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens",
                xticklabels=sorted(df["species"].unique()),
                yticklabels=sorted(df["species"].unique()))
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig("iris_confusion_matrix.png", dpi=150)
    plt.close()

    # Feature importance
    importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values()
    plt.figure(figsize=(6, 4))
    importances.plot(kind="barh", color="teal")
    plt.title("Feature Importance")
    plt.tight_layout()
    plt.savefig("iris_feature_importance.png", dpi=150)
    plt.close()

    print("Saved plots: iris_confusion_matrix.png, iris_feature_importance.png")
    return model, scaler


def main():
    df = load_data()
    explore_data(df)
    train_and_evaluate(df)


if __name__ == "__main__":
    main()