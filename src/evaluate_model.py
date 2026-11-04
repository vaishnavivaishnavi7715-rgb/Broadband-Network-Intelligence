from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
)
ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "broadband_telemetry.csv"
MODEL_PATH = ROOT / "models" / "broadband_anomaly_model.joblib"
REPORT_DIR = ROOT / "reports"
FEATURES = [
    "latency_ms",
    "download_mbps",
    "upload_mbps",
    "packet_loss_pct",
    "signal_strength_dbm",
    "connection_drops",
]
TARGET = "is_anomaly"
def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA_PATH)
    required = FEATURES + [TARGET]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    df = df[required].dropna()
    X = df[FEATURES]
    y = df[TARGET].astype(int)
    if not set(y.unique()).issubset({0, 1}) or y.nunique() != 2:
        raise ValueError("is_anomaly must contain both labels 0 and 1.")
    print("Dataset rows:", len(df))
    print("\nLabel distribution:")
    print(y.value_counts().sort_index())
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    saved_model = joblib.load(MODEL_PATH)
    if isinstance(saved_model, dict):
        model = saved_model.get("model")
        saved_features = saved_model.get("features")
        if model is None:
            raise ValueError("Saved model dictionary has no 'model' entry.")
        if saved_features is not None and list(saved_features) != FEATURES:
            raise ValueError(
                f"Model features mismatch. Saved: {saved_features}; "
                f"expected: {FEATURES}"
            )
    else:
        model = saved_model
    if not hasattr(model, "predict"):
        raise TypeError(f"Loaded object has no predict method: {type(model)}")
    if hasattr(model, "feature_names_in_"):
        expected = list(model.feature_names_in_)
        if expected != FEATURES:
            raise ValueError(
                f"Model expects {expected}, but evaluation uses {FEATURES}"
            )
    predictions = pd.Series(
        model.predict(X_test), index=y_test.index
    ).astype(int)
    if not set(predictions.unique()).issubset({0, 1}):
        raise ValueError("Model returned labels other than 0 and 1.")
    metrics = {
        "evaluation_type": (
            "Diagnostic holdout; independence from training data unverified"
        ),
        "dataset": str(DATA_PATH),
        "model": str(MODEL_PATH),
        "rows_used": int(len(df)),
        "test_rows": int(len(y_test)),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(
            precision_score(y_test, predictions, zero_division=0)
        ),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1_score": float(f1_score(y_test, predictions, zero_division=0)),
        "confusion_matrix_labels": ["normal_0", "anomaly_1"],
        "confusion_matrix": confusion_matrix(
            y_test, predictions, labels=[0, 1]
        ).tolist(),
    }
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X_test)
        classes = list(model.classes_)
        if 1 in classes:
            metrics["roc_auc"] = float(
                roc_auc_score(y_test, probabilities[:, classes.index(1)])
            )
    report_text = classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["Normal", "Anomaly"],
        zero_division=0,
    )
    with open(REPORT_DIR / "model_evaluation.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    with open(
        REPORT_DIR / "classification_report.txt", "w", encoding="utf-8"
    ) as f:
        f.write("Broadband Anomaly Model Evaluation\n")
        f.write("=" * 38 + "\n\n")
        f.write(
            "WARNING: Diagnostic holdout; training-data independence "
            "is unverified.\n\n"
        )
        f.write(json.dumps(metrics, indent=2))
        f.write("\n\nClassification report:\n")
        f.write(report_text)
    cm = confusion_matrix(y_test, predictions, labels=[0, 1])
    pd.DataFrame(
        cm,
        index=["Actual Normal", "Actual Anomaly"],
        columns=["Predicted Normal", "Predicted Anomaly"],
    ).to_csv(REPORT_DIR / "confusion_matrix.csv")
    print("\n=== MODEL EVALUATION ===")
    for name, value in metrics.items():
        print(f"{name}: {value}")
    print("\n=== CLASSIFICATION REPORT ===")
    print(report_text)
    print("\nReports saved in:", REPORT_DIR)
if __name__ == "__main__":
    main()
