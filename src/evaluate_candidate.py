import json
from pathlib import Path
import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "broadband_candidate_telemetry.csv"
MODEL_PATH = ROOT / "models" / "broadband_anomaly_model.joblib"
REPORT_DIR = ROOT / "reports"
df = pd.read_csv(DATA_PATH)
saved = joblib.load(MODEL_PATH)
if isinstance(saved, dict):
    model = saved["model"]
    features = saved.get("features")
else:
    model = saved
    features = getattr(model, "feature_names_in_", None)
if not features:
    features = [
        "latency_ms",
        "download_mbps",
        "upload_mbps",
        "packet_loss_pct",
        "signal_strength_dbm",
        "connection_drops",
    ]
features = list(features)
if "is_anomaly" not in df.columns:
    raise ValueError("Candidate dataset is missing the is_anomaly label.")
missing = [col for col in features if col not in df.columns]
if missing:
    raise ValueError(f"Candidate dataset is missing model features: {missing}")
if df[features + ["is_anomaly"]].isnull().any().any():
    raise ValueError("Missing values found. Evaluation stopped safely.")
X = df[features]
y = df["is_anomaly"].astype(int)
if not set(y.unique()).issubset({0, 1}):
    raise ValueError("Labels must be binary: 0 and 1.")
pred = model.predict(X).astype(int)
prob = None
if hasattr(model, "predict_proba"):
    probabilities = model.predict_proba(X)
    classes = list(model.classes_)
    if 1 in classes:
        prob = probabilities[:, classes.index(1)]
results = {
    "evaluation_type": "Existing active model on separate candidate synthetic dataset",
    "dataset": DATA_PATH.name,
    "dataset_rows": int(len(df)),
    "features_used": features,
    "actual_label_counts": {
        str(k): int(v) for k, v in y.value_counts().sort_index().items()
    },
    "accuracy": float(accuracy_score(y, pred)),
    "precision": float(precision_score(y, pred, zero_division=0)),
    "recall": float(recall_score(y, pred, zero_division=0)),
    "f1_score": float(f1_score(y, pred, zero_division=0)),
    "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1]).tolist(),
    "warning": (
        "Candidate data is synthetic. Results do not establish performance "
        "on real broadband networks. Confirm dataset independence and label quality."
    ),
}
if prob is not None and y.nunique() == 2:
    results["roc_auc"] = float(roc_auc_score(y, prob))
REPORT_DIR.mkdir(parents=True, exist_ok=True)
json_path = REPORT_DIR / "candidate_external_evaluation.json"
text_path = REPORT_DIR / "candidate_external_classification_report.txt"
matrix_path = REPORT_DIR / "candidate_external_confusion_matrix.csv"
json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
text_path.write_text(
    classification_report(
        y, pred, labels=[0, 1],
        target_names=["Normal", "Anomaly"],
        zero_division=0,
    ),
    encoding="utf-8",
)
pd.DataFrame(
    results["confusion_matrix"],
    index=["Actual_Normal", "Actual_Anomaly"],
    columns=["Predicted_Normal", "Predicted_Anomaly"],
).to_csv(matrix_path)
print("\n=== CANDIDATE DATASET EVALUATION ===")
for key, value in results.items():
    print(f"{key}: {value}")
print("\n=== CLASSIFICATION REPORT ===")
print(text_path.read_text(encoding="utf-8"))
print(f"Reports saved in: {REPORT_DIR}")
