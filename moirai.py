import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
import sys
from einops import rearrange
from gluonts.dataset.multivariate_grouper import MultivariateGrouper
from gluonts.dataset.pandas import PandasDataset
from gluonts.dataset.split import split

from uni2ts.eval_util.plot import plot_single, plot_next_multi
from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
from uni2ts.model.moirai_moe import MoiraiMoEForecast, MoiraiMoEModule

MODEL = "moirai-moe"  # model name: choose from {'moirai', 'moirai-moe'}

SIZE = "small"  # model size: choose from {'small', 'base', 'large'}
PDT = 24  # prediction length: any positive integer
CTX = 24*7  # context length: any positive integer
PSZ = "auto"  # patch size: choose from {"auto", 8, 16, 32, 64, 128}
BSZ = 32  # batch size: any positive integer
TEST = 24*7  # test set length: any positive integer

# Load dataframe
# url = (
#     "https://gist.githubusercontent.com/rsnirwan/a8b424085c9f44ef2598da74ce43e7a3"
#     "/raw/b6fdef21fe1f654787fa0493846c546b7f9c4df2/ts_long.csv"
# )
# filename='modified_1.csv'
# df = pd.read_csv("../../cht_mod/"+filename, index_col=0, parse_dates=True)

if len(sys.argv) != 2:
    print("Usage: python predict.py <filename>")
    sys.exit(1)

input_filename = sys.argv[1]
df = pd.read_csv(input_filename, index_col=0, parse_dates=True)


initial_entry_count = len(df)

df_head = df.head()
df["item_id"]=0


initial_entry_count, df_head
df = df[~df.index.duplicated(keep='first')]



# Resample the data to a frequency of 2T (2 minutes), forward-filling missing values

df = df.resample('H').ffill()



# Check the final number of entries and inspect the data

final_entry_count = len(df)

df_head_processed = df.head()



final_entry_count, df_head_processed

# Convert into GluonTS dataset
ds = PandasDataset.from_long_dataframe(df, target="value", item_id="item_id")

# Split into train/test set
train, test_template = split(
    ds, offset=-TEST
)  # assign last TEST time steps as test set

# Construct rolling window evaluation
test_data = test_template.generate_instances(
    prediction_length=PDT,  # number of time steps for each prediction
    windows=TEST // PDT,  # number of windows in rolling window evaluation
    distance=PDT,  # number of time steps between each window - distance=PDT for non-overlapping windows
)

# Prepare model
if MODEL == "moirai":
    model = MoiraiForecast(
        module=MoiraiModule.from_pretrained(f"Salesforce/moirai-1.1-R-small"),
        prediction_length=PDT,
        context_length=CTX,
        patch_size=PSZ,
        num_samples=100,
        target_dim=1,
        feat_dynamic_real_dim=ds.num_feat_dynamic_real,
        past_feat_dynamic_real_dim=ds.num_past_feat_dynamic_real,
    )
elif MODEL == "moirai-moe":
    model = MoiraiMoEForecast(
        module=MoiraiMoEModule.from_pretrained(f"Salesforce/moirai-moe-1.0-R-small"),
        prediction_length=PDT,
        context_length=CTX,
        patch_size=16,
        num_samples=100,
        target_dim=1,
        feat_dynamic_real_dim=ds.num_feat_dynamic_real,
        past_feat_dynamic_real_dim=ds.num_past_feat_dynamic_real,
    )

predictor = model.create_predictor(batch_size=BSZ)
forecasts = predictor.predict(test_data.input)

input_it = iter(test_data.input)
label_it = iter(test_data.label)
forecast_it = iter(forecasts)

prediction_results = []

# Extract forecasts and align timestamps with predictions
for forecast in forecast_it:
    start_date = pd.Period(forecast.start_date).to_timestamp()
    pred_length = len(forecast.mean)  # Number of predicted time steps
    timestamps = pd.date_range(start=start_date, periods=pred_length, freq="H")
    
    # Collect timestamp and predicted mean
    for timestamp, pred_value in zip(timestamps, forecast.mean):
        prediction_results.append({"time": timestamp, "predicted_value": pred_value})

print("Reached here")
# Convert predictions to a DataFrame
pred_df = pd.DataFrame(prediction_results)

# Reset index of the original DataFrame for merging
df.reset_index(inplace=True)
df.rename(columns={"index": "time"}, inplace=True)
df = df.loc[:, ~df.columns.duplicated()]
# Merge the original data with predictions
merged_df = pd.merge(df, pred_df, on="time", how="outer")

# Sort by timestamp to ensure correct ordering
merged_df.sort_values(by="time", inplace=True)

# Save the merged DataFrame to a CSV file
merged_df.to_csv("predicted_data.csv", index=False)

print("Predicted data has been saved to 'predicted_data.csv'.")

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def analyze_errors(df, actual_col='value', pred_col='predicted_value', n_last=TEST):
    # Get last n entries
    df_subset = df.tail(n_last)
    
    actual = df_subset[actual_col]
    pred = df_subset[pred_col]
    
    # Calculate various error metrics
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    
    # Calculate SMAPE
    smape = 100 * np.mean(np.abs(pred - actual) / ((np.abs(actual) + np.abs(pred)) / 2))
    
    # Calculate MAPE
    mape = 100 * np.mean(np.abs((actual - pred) / actual))
    
    # Calculate basic statistics
    mean_error = np.mean(pred - actual)
    std_error = np.std(pred - actual)
    
    # Count the number of underpredictions
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

# Read the CSV file
# print("Original Dataset without input Tweaking")
df = pd.read_csv('predicted_data.csv')

# Get error analysis
results = analyze_errors(df)

# Print results
for metric, value in results.items():
    print(f"{metric}: {value:.4f}" if isinstance(value, float) else f"{metric}: {value}")