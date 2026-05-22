import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")

# 1. Generate Synthetic Historical Data (Replace with your dataset path)
# e.g., df = pd.read_csv('traffic_data.csv', parse_dates=['timestamp'], index_col='timestamp')
dates = pd.date_range(start="2026-01-01", periods=200, freq="H")
traffic_volume = (
    100
    + 30 * np.sin(np.arange(200) * (2 * np.pi / 24))
    + np.random.normal(0, 10, 200)
)
df = pd.DataFrame(data={"traffic": traffic_volume}, index=dates)
df.index.freq = "H"

# 2. Preprocess & Clean Data
df["traffic"] = df["traffic"].fillna(method="ffill")

# Split into Train and Test sets (80/20)
train_size = int(len(df) * 0.8)
train, test = df.iloc[:train_size], df.iloc[train_size:]

# 3. Build and Train Time-Series Model (SARIMAX)
# (p,d,q) x (P,D,Q,s) -> Configured for daily seasonality (24 hours)
model = SARIMAX(
    train["traffic"], order=(1, 1, 1), seasonal_order=(1, 1, 1, 24)
)
model_fit = model.fit(disp=False)

# 4. Forecast Future Trends
predictions = model_fit.predict(start=test.index[0], end=test.index[-1])
predictions.index = test.index

# 5. Evaluate Model Accuracy
mae = mean_absolute_error(test["traffic"], predictions)
rmse = np.sqrt(mean_squared_error(test["traffic"], predictions))

print("--- Model Evaluation Metrics ---")
print(f"Mean Absolute Error (MAE)   : {mae:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

# 6. Visualize Predictions
plt.figure(figsize=(12, 6))
plt.plot(train.index[-48:], train["traffic"][-48:], label="Historical (Last 48h)")
plt.plot(test.index, test["traffic"], label="Actual Future Trends", color="gray")
plt.plot(
    test.index,
    predictions,
    label="Forecasted Trends",
    color="red",
    linestyle="--",
)
plt.title("Predictive Analytics: Historical vs Forecasted Trends")
plt.xlabel("Timestamp")
plt.ylabel("Value / Volume")
plt.legend()
plt.grid(True)
plt.savefig("predictive_forecast.png")  # Saves the visualization
plt.show()