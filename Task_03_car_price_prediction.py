import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ------------------------------------------------------------------
# 1. LOAD DATA
# ------------------------------------------------------------------
df = pd.read_csv("car_data.csv")

print("Shape:", df.shape)
print("\nFirst 5 rows:\n", df.head())
print("\nMissing values:\n", df.isnull().sum())
print("\nFuel_Type values:", df["Fuel_Type"].unique())
print("Selling_type values:", df["Selling_type"].unique())
print("Transmission values:", df["Transmission"].unique())

# ------------------------------------------------------------------
# 2. EDA
# ------------------------------------------------------------------
plt.figure(figsize=(6, 5))
sns.scatterplot(data=df, x="Present_Price", y="Selling_Price", hue="Fuel_Type")
plt.title("Present Price vs Selling Price")
plt.tight_layout()
plt.savefig("present_vs_selling_price.png")
plt.close()

# ------------------------------------------------------------------
# 3. FEATURE ENGINEERING
# ------------------------------------------------------------------
current_year = 2026
df["Car_Age"] = current_year - df["Year"]
df = df.drop(columns=["Year", "Car_Name"])  # Car_Name is high-cardinality, not predictive here

# One-hot encode categoricals
df = pd.get_dummies(df, columns=["Fuel_Type", "Selling_type", "Transmission"], drop_first=True)

plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(numeric_only=True), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig("correlation_heatmap.png")
plt.close()

# ------------------------------------------------------------------
# 4. TRAIN / TEST SPLIT
# ------------------------------------------------------------------
X = df.drop(columns=["Selling_Price"])
y = df["Selling_Price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ------------------------------------------------------------------
# 5. TRAIN MODELS
# ------------------------------------------------------------------
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)

rf_model = RandomForestRegressor(n_estimators=200, random_state=42)
rf_model.fit(X_train, y_train)

# ------------------------------------------------------------------
# 6. EVALUATE
# ------------------------------------------------------------------
def evaluate(name, y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"\n{name}")
    print(f"  R2 Score : {r2:.4f}")
    print(f"  MAE      : {mae:.4f} lakhs")
    print(f"  RMSE     : {rmse:.4f} lakhs")

lr_pred = lr_model.predict(X_test_scaled)
rf_pred = rf_model.predict(X_test)

evaluate("Linear Regression", y_test, lr_pred)
evaluate("Random Forest Regressor", y_test, rf_pred)

# ------------------------------------------------------------------
# 7. FEATURE IMPORTANCE
# ------------------------------------------------------------------
importances = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=False)

plt.figure(figsize=(8, 5))
importances.plot(kind="barh")
plt.title("Feature Importance (Random Forest)")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.close()

print("\nFeature importances:\n", importances)

# ------------------------------------------------------------------
# 8. ACTUAL VS PREDICTED
# ------------------------------------------------------------------
plt.figure(figsize=(6, 6))
plt.scatter(y_test, rf_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
plt.xlabel("Actual Selling Price (lakhs)")
plt.ylabel("Predicted Selling Price (lakhs)")
plt.title("Actual vs Predicted Price (Random Forest)")
plt.tight_layout()
plt.savefig("actual_vs_predicted.png")
plt.close()

print("\nDone. Saved: present_vs_selling_price.png, correlation_heatmap.png, feature_importance.png, actual_vs_predicted.png")
