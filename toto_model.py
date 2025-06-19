

import os, sys

# 1) Where this script lives
HERE = os.path.abspath(os.path.dirname(__file__))

# 2) Build the path to your nested repo code
#    (adjust if you only have one level of `toto/`, or two levels)
NESTED = os.path.join(HERE, "toto")

# 3) Prepend it to sys.path so your imports resolve there
if os.path.isdir(NESTED):
    sys.path.insert(0, NESTED)
else:
    raise RuntimeError(f"Expected code under {NESTED}, but didn't find it")


import matplotlib
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import torch

from data.util.dataset import MaskedTimeseries
from inference.forecaster import TotoForecaster
from model.toto import Toto

if len(sys.argv) != 2:
    print("Usage: python predict.py <filename>")
    sys.exit(1)
input_filename = sys.argv[1]


df = (
    pd.read_csv(input_filename)
    .assign(date=lambda df: pd.to_datetime(df["time"]))
    .assign(timestamp_seconds=lambda df: (df.date - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s'))
)

df

# These lines make gpu execution in CUDA deterministic
# os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
# torch.use_deterministic_algorithms(True)

context_length = 69*24
prediction_length = 24*7

feature_columns = ["value"]
n_variates = len(feature_columns)
interval = 60 * 60  # 15-min intervals
input_df = df.iloc[-(context_length+prediction_length):-prediction_length]
target_df = df.iloc[-prediction_length:]
DEVICE = "cpu"

input_series = torch.from_numpy(input_df[feature_columns].values.T).to(torch.float).to(DEVICE)
input_series.shape

timestamp_seconds = torch.from_numpy(input_df.timestamp_seconds.values.T).expand((n_variates, context_length)).to(input_series.device)
time_interval_seconds=torch.full((n_variates,), interval).to(input_series.device)
start_timestamp_seconds = timestamp_seconds[:, 0]

inputs = MaskedTimeseries(
    series=input_series,
    # The padding mask should be the same shape as the input series.
    # It should be 0 to indicate padding and 1 to indicate valid values.
    padding_mask=torch.full_like(input_series, True, dtype=torch.bool),
    # The ID mask is used for packing unrelated time series along the Variate dimension.
    # This is used in training, and can also be useful for large-scale batch inference in order to
    # process time series of different numbers of variates using batches of a fixed shape.
    # The ID mask controls the channel-wise attention; variates with different IDs cannot attend to each other.
    # If you're not using packing, just set this to zeros.
    id_mask=torch.zeros_like(input_series),
    # As mentioned above, these timestamp features are not currently used by the model;
    # however, they are reserved for future releases.
    timestamp_seconds=timestamp_seconds,
    time_interval_seconds=time_interval_seconds,
)
toto = Toto.from_pretrained('Datadog/Toto-Open-Base-1.0')
toto.to(DEVICE)

# Optionally enable Torch's JIT compilation to speed up inference. This is mainly
# helpful if you want to perform repeated inference, as the JIT compilation can
# take time to wrm up.
toto.compile()

forecaster = TotoForecaster(toto.model)
forecast = forecaster.forecast(
    inputs,
    # We can set any number of timesteps into the future that we'd like to forecast. Because Toto is an autoregressive model,
    # the inference time will be longer for longer forecasts. 
    prediction_length=prediction_length,
    # TOTOForecaster draws samples from a predicted parametric distribution. The more samples, the more stable and accurate the prediction.
    # This is especially important if you care about accurate prediction intervals in the tails.
    # Toto's evaluations were performed using 256 samples. Set this according to your compute budget.
    num_samples=256,
    # TOTOForecaster also handles batching the samples in order to control memory usage.
    # Set samples_per_batch as high as you can without getting OOMs for maximum performance.
    # If you're doing batch inference, the effective batch size sent to the model is (batch_size x samples_per_batch).
    # In this notebook, we're doing unbatched inference, so the effective batch size is samples_per_batch.
    samples_per_batch=256,
    # KV cache should significantly speed up inference, and in most cases should reduce memory usage too.
    use_kv_cache=True,
)

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Get ground truth and predictions as numpy arrays
feature = feature_columns[0]  # e.g., "value"

y_true = target_df[feature].values
y_pred = np.median(forecast.samples.squeeze().cpu().numpy(), axis=-1)

# RMSE
rmse = np.sqrt(mean_squared_error(y_true, y_pred))

# MAE
mae = mean_absolute_error(y_true, y_pred)

# R-squared
r2 = r2_score(y_true, y_pred)

# SMAPE
def smape(y_true, y_pred):
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-8))
smape_val = smape(y_true, y_pred)

# MAPE
def mape(y_true, y_pred):
    return 100 * np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8)))
mape_val = mape(y_true, y_pred)

# Mean Error
mean_error = np.mean(y_pred - y_true)

# Std Error
std_error = np.std(y_pred - y_true)

# Under/Over Predictions
under_predictions = np.sum(y_pred < y_true)
over_predictions = np.sum(y_pred > y_true)

# Print metrics
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"R-squared: {r2:.4f}")
print(f"SMAPE: {smape_val:.4f}")
print(f"MAPE: {mape_val:.4f}")
print(f"Mean Error: {mean_error:.4f}")
print(f"Std Error: {std_error:.4f}")
print(f"Under Predictions: {under_predictions}")
print(f"Over Predictions: {over_predictions}")

