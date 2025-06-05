import pandas as pd
import torch
from chronos import BaseChronosPipeline
import sys

# Load local CSV file

if len(sys.argv) != 2:
    print("Usage: python predict.py <filename>")
    sys.exit(1)

file_path = sys.argv[1]
# file_path = "./../my_work_2/4.csv"  # Replace with your actual file path
df = pd.read_csv(file_path, parse_dates=["time"], index_col="time")  # Ensure your timestamp column name matches

# Ensure the data is in 2-minute frequency
# df = df.resample("H").mean()  # Resample to 2-minute intervals (adjust method if necessary)
# df.dropna(inplace=True)  # Remove any missing values after resampling

# Chronos Model
pipeline = BaseChronosPipeline.from_pretrained(
    "amazon/chronos-t5-small",  
    device_map="cpu",  
    torch_dtype=torch.bfloat16,
)

# Define parameters
PDT = 24*7  # Prediction length
CTX = 48 * 7  # Context length
PSZ = "auto"  # Patch size
TEST = 24 * 7  # Test set length

# Convert to tensor
context_data = torch.tensor(df["value"].values, dtype=torch.float32)  # Replace with your actual column name

# Predict
quantiles, mean = pipeline.predict_quantiles(
    context=context_data[-CTX:],  # Use the last CTX values
    prediction_length=PDT,
    quantile_levels=[0.1, 0.5, 0.9],
)

actual_values = df["value"].iloc[-PDT:].values

last_timestamp = df.index[-1]
forecast_timestamps = pd.date_range(start=last_timestamp + pd.Timedelta(hours=1), periods=PDT, freq="H")

# Print forecast timestamps
print(forecast_timestamps)

# Create a DataFrame to store predictions
forecast_df = pd.DataFrame({
    "time": forecast_timestamps,
    "low": quantiles[0, :, 0].tolist(),
    "median": quantiles[0, :, 1].tolist(),
    "high": quantiles[0, :, 2].tolist(),
    
})

from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Assuming for a particular prediction window you have:
# - actual_values: a numpy array of actual observed values for the prediction window
# - pred_median: a torch tensor containing the median predictions

# Convert the median predictions to a numpy array if needed:
median_predictions = quantiles.numpy()

# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(actual_values, quantiles[0, :, 1])

# Calculate Root Mean Squared Error (RMSE)
rmse = np.sqrt(mse)

# Calculate R² (coefficient of determination)
r2 = r2_score(actual_values, quantiles[0, :, 1])

print("MSE:", mse)
print("RMSE:", rmse)
print("R^2:", r2)

# Assuming you already have:
# - actual_values: a 1D NumPy array of shape (PDT,)
# - median_predictions: a 1D NumPy array of shape (PDT,)

# Count under and over predictions
under_predictions = np.sum(quantiles[0, :, 1].numpy() < actual_values)
over_predictions = np.sum(quantiles[0, :, 1].numpy() > actual_values)
equal_predictions = np.sum(quantiles[0, :, 1].numpy() == actual_values)  # optional

print(f"Under-predictions: {under_predictions}")
print(f"Over-predictions: {over_predictions}")
print(f"Exact matches   : {equal_predictions}")
