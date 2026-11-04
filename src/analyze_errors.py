from pathlib import Path
import joblib
import pandas as pd
from sklearn.metrics import confusion_matrix
ROOT = Path(__file__).resolve().parent.parent
df = pd.read_csv(ROOT / "data" / "broadband_candidate_telemetry.csv")
saved = joblib.load(ROOT / "models" / "broadband_anomaly_model.joblib")
if isinstance(saved, dict):
    model = saved["model"]
    features = saved.get("features")
else:
    model = saved
    features = getattr(model, "feature_names_in_", None)
if not features:
    features = [
        "latency_ms", "download_mbps", "upload_mbps",
        "packet_loss_pct", "signal_strength_dbm", "connection_drops"
    ]
features = list(features)
actual = df["is_anomaly"].astype(int)
predicted = model.predict(df[features]).astype(int)
errors = df.copy()
errors["predicted_anomaly"] = predicted
errors["error_type"] = "Correct"
errors.loc[(actual == 0) & (predicted == 1), "error_type"] = "False Positive"
errors.loc[(actual == 1) & (predicted == 0), "error_type"] = "False Negative"
mistakes = errors[errors["error_type"] != "Correct"].copy()
out = ROOT / "reports" / "candidate_misclassified_records.csv"
mistakes.to_csv(out, index=False)
print("Confusion matrix:")
print(confusion_matrix(actual, predicted, labels=[0, 1]))
print("\nError counts:")
print(mistakes["error_type"].value_counts())
print("\nAverage metrics by error type:")
print(mistakes.groupby("error_type")[features].mean().round(2))
print(f"\nMisclassified records saved to: {out}")
