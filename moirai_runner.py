import pandas as pd
import numpy as np
import torch
from gluonts.dataset.pandas import PandasDataset
from gluonts.dataset.split import split
from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
from uni2ts.model.moirai_moe import MoiraiMoEForecast, MoiraiMoEModule


def run_moirai(
    csv_path: str,
    ctx_length: int,
    pred_length: int,
    batch_size: int = 32,
    model_size: str = "small",  # kept for compatibility; hardcoded
    patch_size: str = "auto",   # used only in Moirai
    model_name: str = "moirai-moe",
    output_path: str = "predicted_data.csv"
) -> pd.DataFrame:
    # Load and preprocess CSV
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    df = df[~df.index.duplicated(keep="first")]
    df = df.resample("H").ffill()
    df["item_id"] = 0

    ds = PandasDataset.from_long_dataframe(df, target="value", item_id="item_id")

    # Split into train and test
    TEST = pred_length
    train, test_template = split(ds, offset=-TEST)
    test_data = test_template.generate_instances(
        prediction_length=pred_length,
        windows=TEST // pred_length,
        distance=pred_length
    )
    print("start")
    # Load model using fixed IDs (no dynamic model_size!)
    if model_name == "moirai":
        model = MoiraiForecast(
            module=MoiraiModule.from_pretrained("Salesforce/moirai-1.1-R-small"),
            prediction_length=pred_length,
            context_length=ctx_length,
            patch_size=patch_size if patch_size != "auto" else None,
            num_samples=100,
            target_dim=1,
            feat_dynamic_real_dim=ds.num_feat_dynamic_real,
            past_feat_dynamic_real_dim=ds.num_past_feat_dynamic_real,
        )
    elif model_name == "moirai-moe":
        model = MoiraiMoEForecast(
            module=MoiraiMoEModule.from_pretrained("Salesforce/moirai-moe-1.0-R-small"),
            prediction_length=pred_length,
            context_length=ctx_length,
            patch_size=16,
            num_samples=100,
            target_dim=1,
            feat_dynamic_real_dim=ds.num_feat_dynamic_real,
            past_feat_dynamic_real_dim=ds.num_past_feat_dynamic_real,
        )
    else:
        raise ValueError("model_name must be 'moirai' or 'moirai-moe'")

    # Predict
    predictor = model.create_predictor(batch_size=batch_size)
    forecasts = predictor.predict(test_data.input)

    prediction_results = []
    for forecast in forecasts:
        start_date = pd.Period(forecast.start_date).to_timestamp()
        pred_len = len(forecast.mean)
        timestamps = pd.date_range(start=start_date, periods=pred_len, freq="H")
        for timestamp, pred_value in zip(timestamps, forecast.mean):
            prediction_results.append({"time": timestamp, "predicted_value": pred_value})

    pred_df = pd.DataFrame(prediction_results)

    # Merge with original
    df.reset_index(inplace=True)
    df.rename(columns={"index": "time"}, inplace=True)
    df["time"] = pd.to_datetime(df["time"])
    pred_df["time"] = pd.to_datetime(pred_df["time"])

    merged = pd.merge(df[["time", "value"]], pred_df, on="time", how="outer").sort_values("time")
    merged.to_csv(output_path, index=False)

    return merged
