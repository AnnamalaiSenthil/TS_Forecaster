from runner import forecast_with_metrics

df, metrics = forecast_with_metrics(
    csv_path="1.csv",
    model="chronos",         # or "moirai", "chronos"
    ctx_length=168,
    pred_length=24,
    batch_size=32,
    model_size="base",
    patch_size="auto",
    output_path="output_toto.csv"
)

print("✅ Forecast completed!\n")
print("📊 Metrics:")
for k, v in metrics.items():
    print(f"{k}: {v:.4f}")

print("\n📈 Predictions (last 5 rows):")
print(df.tail())
