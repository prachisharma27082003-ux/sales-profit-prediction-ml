import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
df = pd.read_csv("data/sales_data.csv")

# Create Revenue, Cost and Profit
df["Revenue"] = df["Units_Sold"] * df["Selling_Price"]
df["Cost"] = df["Units_Sold"] * df["Cost_Price"]
df["Profit"] = df["Revenue"] - df["Cost"]

# Convert Date to datetime and create Year/Month
df["Date"] = pd.to_datetime(df["Date"])
df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month

# Features and Target
feature_cols = ["Units_Sold", "Selling_Price", "Cost_Price", "Marketing_Spend", "Month", "Year"]
X = df[feature_cols]
y = df["Profit"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Linear Regression model
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
y_pred_lr = lin_reg.predict(X_test)

# Random Forest model
rf = RandomForestRegressor(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# Evaluate models
print("📌 Linear Regression:")
print("MAE:", mean_absolute_error(y_test, y_pred_lr))
print("MSE:", mean_squared_error(y_test, y_pred_lr))
print("R2 :", r2_score(y_test, y_pred_lr))

print("\n🌲 Random Forest:")
print("MAE:", mean_absolute_error(y_test, y_pred_rf))
print("MSE:", mean_squared_error(y_test, y_pred_rf))
print("R2 :", r2_score(y_test, y_pred_rf))

# Feature importance
importances = rf.feature_importances_
feat_imp = pd.DataFrame({"Feature": feature_cols, "Importance": importances}) \
    .sort_values(by="Importance", ascending=False)

print("\nFeature Importance:")
print(feat_imp)
