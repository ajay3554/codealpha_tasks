# CodeAlpha_IrisFlowerClassification

**Task 1 - Data Science Internship, CodeAlpha**

Classifies Iris flowers (setosa, versicolor, virginica) using sepal/petal
measurements with a RandomForestClassifier.

## How to run
```
pip install pandas scikit-learn matplotlib seaborn
python iris_classification.py
```

This repo already includes the real `Iris.csv` dataset (150 rows, Kaggle
format) - the script auto-detects and uses it. If the file is ever
missing, it falls back to scikit-learn's built-in Iris dataset (same data).

## What it does
- EDA: summary stats, pairplot, correlation heatmap
- Trains/evaluates a RandomForestClassifier
- Saves confusion matrix and feature importance plots

## Results (on the real dataset)
- Accuracy: **90%**
- Setosa: 100% precision/recall (perfectly separable)
- Versicolor / Virginica: some overlap, ~82-89% precision

## Output
Console metrics (accuracy, classification report) + PNG plots in this folder.
