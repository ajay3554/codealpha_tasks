# CodeAlpha_SalesPrediction

**Task 4 - Data Science Internship, CodeAlpha**

Predicts sales based on advertising spend across TV, Radio, and Newspaper.

## How to run
```
pip install pandas scikit-learn matplotlib seaborn
python sales_prediction.py
```

This repo already includes the real `Advertising.csv` dataset (200 rows) -
the script auto-detects and uses it. If the file is ever missing, a
realistic synthetic dataset (same columns) is generated instead so the
script still runs end-to-end.

## What it does
- EDA: correlation heatmap, channel-wise scatter plots
- Trains/compares LinearRegression vs RandomForestRegressor
- Shows each channel's impact on sales (regression coefficients)
- Reports MAE, RMSE, R2 and plots actual vs predicted sales

## Results (on the real dataset)
| Model | MAE | RMSE | R2 |
|---|---|---|---|
| Linear Regression | 1.461 | 1.782 | 0.899 |
| Random Forest Regressor | 0.614 | 0.741 | **0.983** |

TV has the strongest impact on sales, followed by Radio; Newspaper spend
has almost no measurable effect.

## Output
Console metrics + PNG plots in this folder.
