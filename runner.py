from moirai_runner import run_moirai
from chronos_runner import run_chronos
from toto.toto_runner import run_toto

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import pandas as pd


def forecast_with_metrics(
    csv_path: str,
    model: str = "moirai",
    ctx_length: int = 168,
    pred_length: int = 24,
    batch_size: int = 32,
    model_size: str = "small",
    patch_size: str = "auto",
    output_path: str = "predicted_data.csv"
) -> (pd.DataFrame, dict):
    model_map = {
        "moirai": run_moirai,
        "moirai-moe": run_moirai,
        "chronos": run_chronos,
        "toto": run_toto,
    }

    if model not in model_map:
        raise ValueError(f"Unknown model: {model}")

    df = model_map[model](
        csv_path=csv_path,
        ctx_length=ctx_length,
        pred_length=pred_length,
        batch_size=batch_size,
        model_size=model_size,
        patch_size=patch_size,
        model_name=model,  # for moirai variant detection
        output_path=output_path,
    )

    # Drop NaNs to compute metrics
    print("Last 10 rows of merged DF:")
    print(df.tail(10))

    print("\nNaN counts:")
    print(df[["value", "predicted_value"]].isna().sum())

    print("\nMerged shape before dropna:", df.shape)
    df = df.dropna(subset=["value", "predicted_value"])
    print("Merged shape after dropna:", df.shape)

    actual = df["value"].values
    pred = df["predicted_value"].values

    df = df.dropna(subset=["value", "predicted_value"])
    actual = df["value"]
    pred = df["predicted_value"]

    mae = mean_absolute_error(actual, pred)
    rmse = mean_squared_error(actual, pred, squared=False)
    r2 = r2_score(actual, pred)
    smape = 100 * np.mean(np.abs(pred - actual) / ((np.abs(actual) + np.abs(pred)) / 2))
    mape = 100 * np.mean(np.abs((actual - pred) / actual))

    metrics = {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "SMAPE": smape,
        "MAPE": mape,
    }

    return df, metrics
