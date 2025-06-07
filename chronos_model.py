import pandas as pd
import torch
from chronos import BaseChronosPipeline
import sys
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def analyze_errors(actual, pred):
    # Ensure numpy arrays
    actual = np.array(actual)
    pred = np.array(pred)
    
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    smape = 100 * np.mean(np.abs(pred - actual) / ((np.abs(actual) + np.abs(pred)) / 2))
    mape = 100 * np.mean(np.abs((actual - pred) / actual))
    mean_error = np.mean(pred - actual)
    std_error = np.std(pred - actual)
    under_predictions = np.sum(pred < actual)
    over_predictions = np.sum(pred > actual)
    
    results = {
        'RMSE': rmse,
        'MAE': mae,
        'R-squared': r2,
        'SMAPE': smape,
        'MAPE': mape,
        'Mean Error': mean_error,
        'Std Error': std_error,
        'Under Predictions': under_predictions,
        'Over Predictions': over_predictions
    }
    return results

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
# actual_values = actual_values.numpy()

last_timestamp = df.index[-1]
forecast_timestamps = pd.date_range(start=last_timestamp + pd.Timedelta(hours=1), periods=PDT, freq="H")

# Print forecast timestamps
# print(forecast_timestamps)

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
median_predictions = quantiles[0,:,1].numpy()

# -------------------------------------------------

results = analyze_errors(actual_values, median_predictions)

for metric, value in results.items():
    print(f"{metric}: {value:.4f}" if isinstance(value, float) else f"{metric}: {value}")