import pandas as pd
import numpy as np
import torch
from chronos import BaseChronosPipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def run_chronos(
    csv_path: str,
    ctx_length: int,
    pred_length: int,
    batch_size: int = 32,
    model_size: str = "small",
    patch_size: str = "auto",
    model_name: str = "chronos",
    output_path: str = "predicted_data.csv"
) -> pd.DataFrame:
    df = pd.read_csv(csv_path, parse_dates=["time"], index_col="time")
    df = df.sort_index()
    df.index = df.index.floor("h")

    # Load pipeline
    pipeline = BaseChronosPipeline.from_pretrained(
        "amazon/chronos-t5-small",
        device_map="cpu",
        torch_dtype=torch.bfloat16,
    )

    # Predict
    context = torch.tensor(df["value"].values, dtype=torch.float32)[-ctx_length:]
    quantiles, _ = pipeline.predict_quantiles(
        context=context,
        prediction_length=pred_length,
        quantile_levels=[0.1, 0.5, 0.9],
    )

    y_pred = quantiles[0, :, 1].numpy()
    last_time = df.index[-1]
    forecast_timestamps = pd.date_range(start=last_time + pd.Timedelta(hours=1), periods=pred_length, freq="h")

    # Forecast DataFrame
    forecast_df = pd.DataFrame({
        "time": forecast_timestamps,
        "predicted_value": y_pred
    })

    # Pull last true values
    actual_df = df.tail(pred_length).copy().reset_index()
    actual_df.rename(columns={"time": "time"}, inplace=True)

    # Match shape and return expected column names
    result = forecast_df.copy()
    result["value"] = actual_df["value"].values if len(actual_df) == pred_length else np.nan

    result = result[["time", "value", "predicted_value"]]
    result.to_csv(output_path, index=False)
    return result
