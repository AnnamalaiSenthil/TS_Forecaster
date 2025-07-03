# toto.py
import sys
import os

# Add the project root to sys.path if not already there
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from .data.util.dataset import MaskedTimeseries
from .inference.forecaster import TotoForecaster
from .model.toto import Toto



def run_toto(
    csv_path: str,
    ctx_length: int,
    pred_length: int,
    batch_size: int = 32,  # ignored
    model_size: str = "base",  # default to the one that works
    patch_size: str = "auto",  # ignored
    model_name: str = "toto",
    output_path: str = "predicted_data.csv"
) -> pd.DataFrame:
    

    import pandas as pd
    import torch
    import numpy as np
    from pathlib import Path

    from toto.data.util.dataset import MaskedTimeseries
    from toto.inference.forecaster import TotoForecaster
    from toto.model.toto import Toto

    # ✅ Use working repo ID
    model_map = {
        "base": "Datadog/Toto-Open-Base-1.0"
        # You can add more later if needed and tested
    }

    df = pd.read_csv(csv_path)
    df = df.sort_values("time").assign(
        date=pd.to_datetime(df["time"]),
        timestamp_seconds=lambda d: (d["date"] - pd.Timestamp("1970-01-01")) // pd.Timedelta("1s")
    )

    interval = 3600
    input_df = df.iloc[-(ctx_length + pred_length):-pred_length]
    target_df = df.iloc[-pred_length:]
    input_series = torch.from_numpy(input_df[["value"]].values.T).float()
    timestamp_seconds = torch.from_numpy(input_df.timestamp_seconds.values.T).expand((1, ctx_length))
    time_interval_seconds = torch.full((1,), interval)

    inputs = MaskedTimeseries(
        series=input_series,
        padding_mask=torch.ones_like(input_series, dtype=torch.bool),
        id_mask=torch.zeros_like(input_series),
        timestamp_seconds=timestamp_seconds,
        time_interval_seconds=time_interval_seconds,
    )

    # ✅ Use the working model ID only
    model = Toto.from_pretrained(model_map[model_size])
    model.to("cpu").compile()
    forecaster = TotoForecaster(model.model)

    forecast = forecaster.forecast(
        inputs,
        prediction_length=pred_length,
        num_samples=256,
        samples_per_batch=256,
        use_kv_cache=True,
    )

    y_pred = np.median(forecast.samples.squeeze().cpu().numpy(), axis=-1)
    forecast_timestamps = pd.date_range(
        start=input_df["date"].iloc[-1] + pd.Timedelta(hours=1),
        periods=pred_length,
        freq="H"
    )

    pred_df = pd.DataFrame({"time": forecast_timestamps, "predicted_value": y_pred})
    # Ensure both are datetime type
    df["time"] = pd.to_datetime(df["time"])
    pred_df["time"] = pd.to_datetime(pred_df["time"])

    merged = pd.merge(df[["time", "value"]], pred_df, on="time", how="outer").sort_values("time")

    merged.to_csv(output_path, index=False)
    return merged

