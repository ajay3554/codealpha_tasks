"""
CodeAlpha Data Science Internship - Task 4
Sales Prediction using Python
--------------------------------------------------
Predicts sales based on advertising spend across TV, Radio and
Newspaper channels.

NOTE: If you have the original "Advertising.csv" dataset, place it
in this folder and it will be used automatically. Otherwise, a
realistic synthetic dataset (same schema) is generated so the
script runs end-to-end out of the box.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

RANDOM_STATE = 42
DATA_FILE = "Advertising.csv"


def load_data():
    if os.path.exists(DATA_FILE):
        df = pd.read_csv(DATA_FILE)
        if "Unnamed: 0" in df.columns:
            df = df.drop(columns=["Unnamed: 0"])
        print(f"Loaded real dataset: {DATA_FILE}")
        return df

    print(f"'{DATA_FILE}' not found - generating a synthetic dataset with the same schema.")
    rng = np.random.default_rng(RANDOM_STATE)
    n = 200

    tv = np.round(rng.uniform(0, 300, n), 1)
    radio = np.round(rng.uniform(0, 50, n), 1)
    newspaper = np.round(rng.uniform(0, 100, n), 1)

    # TV has the strongest effect, radio moderate, newspaper weak (mirrors the
    # well-known real-world "Advertising" dataset relationships)
    sales = (
        6.0
        + 0.045 * tv
        + 0.19 * radio
        + 0.003 * newspaper
        + rng.normal(0, 1.5, n)
    )
    sales = np.clip(sales, 1, None)

    df = pd.DataFrame({
        "TV": tv,
        "Radio": radio,
        "Newspaper": newspaper,
        "Sales": np.round(sales, 2),
    })
    return df


def explore_data(df):
    print("\n--- Dataset shape ---")
    print(df.shape)
    print("\n--- First 5 rows ---")
    print(df.head())
    print("\n--- Missing values ---")
    print(df.isnull().sum())
    print("\n--- Summary statistics ---")
    print(df.describe())

    plt.figure(figsize=(6, 5))
    sns.heatmap(df.corr(), annot=True, cmap="YlGnBu")
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig("sales_correlation_heatmap.png", dpi=150)
    plt.close()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, col in zip(axes, ["TV", "Radio", "Newspaper"]):
        sns.scatterplot(data=df, x=col, y="Sales", ax=ax)
        ax.set_title(f"{col} vs Sales")
    plt.tight_layout()
    plt.savefig("sales_scatter_plots.png", dpi=150)
    plt.close()

    print("Saved plots: sales_correlation_heatmap.png, sales_scatter_plots.png")


def train_and_evaluate(df):
    X = df[["TV", "Radio", "Newspaper"]]
    y = df["Sales"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "LinearRegression": LinearRegression(),
        "RandomForestRegressor": RandomForestRegressor(n_estimators=300, random_state=RANDOM_STATE),
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)
        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)
        results[name] = {"MAE": mae, "RMSE": rmse, "R2": r2, "model": model, "preds": preds}
        print(f"\n--- {name} ---")
        print(f"MAE:  {mae:.3f}")
        print(f"RMSE: {rmse:.3f}")
        print(f"R2:   {r2:.3f}")

    # Feature impact from linear regression coefficients
    lin_model = results["LinearRegression"]["model"]
    coef_df = pd.Series(lin_model.coef_, index=X.columns).sort_values()
    print("\n--- Advertising channel impact (Linear Regression coefficients) ---")
    print(coef_df)

    plt.figure(figsize=(6, 4))
    coef_df.plot(kind="barh", color="darkorange")
    plt.title("Advertising Channel Impact on Sales")
    plt.tight_layout()
    plt.savefig("sales_channel_impact.png", dpi=150)
    plt.close()

    best_name = max(results, key=lambda k: results[k]["R2"])
    best = results[best_name]
    print(f"\nBest model: {best_name} (R2={best['R2']:.3f})")

    plt.figure(figsize=(6, 5))
    plt.scatter(y_test, best["preds"], alpha=0.7)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
    plt.xlabel("Actual Sales")
    plt.ylabel("Predicted Sales")
    plt.title(f"Actual vs Predicted ({best_name})")
    plt.tight_layout()
    plt.savefig("sales_actual_vs_predicted.png", dpi=150)
    plt.close()

    print("Saved plots: sales_channel_impact.png, sales_actual_vs_predicted.png")
    return results


def main():
    df = load_data()
    explore_data(df)
    train_and_evaluate(df)


if __name__ == "__main__":
    main()