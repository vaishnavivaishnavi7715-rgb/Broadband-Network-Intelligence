from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
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
DATA_PATH = ROOT / "data" / "broadband_telemetry.csv"
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
    df = df[FEATURES + [TARGET]].dropna()
    X = df[FEATURES]
    y = df[TARGET].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )
    print("Training rows:", len(X_train))
    print("Test rows:", len(X_test))
    # This is a separate model; it does NOT replace the active API model.
    evaluation_model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )
    evaluation_model.fit(X_train, y_train)
    predictions = evaluation_model.predict(X_test)
    probabilities = evaluation_model.predict_proba(X_test)[:, 1]
    metrics = {
        "evaluation_type": "80/20 stratified holdout with separate model",
        "dataset_rows": int(len(df)),
        "training_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(
            precision_score(y_test, predictions, zero_division=0)
        ),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1_score": float(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "confusion_matrix": confusion_matrix(
            y_test, predictions, labels=[0, 1]
        ).tolist(),
        "warning": (
            "Holdout evaluation is valid only if duplicate or related records "
            "do not cross the split and the dataset represents the intended "
            "deployment population. Synthetic data limits real-world claims."
        ),
    }
    report = classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["Normal", "Anomaly"],
        zero_division=0,
    )
    with open(
        REPORT_DIR / "leakage_safe_evaluation.json", "w", encoding="utf-8"
    ) as f:
        json.dump(metrics, f, indent=2)
    with open(
        REPORT_DIR / "leakage_safe_classification_report.txt",
        "w",
        encoding="utf-8",
    ) as f:
        f.write(json.dumps(metrics, indent=2))
        f.write("\n\nClassification report:\n")
        f.write(report)
    pd.DataFrame(
        confusion_matrix(y_test, predictions, labels=[0, 1]),
        index=["Actual Normal", "Actual Anomaly"],
        columns=["Predicted Normal", "Predicted Anomaly"],
    ).to_csv(REPORT_DIR / "leakage_safe_confusion_matrix.csv")
    print("\n=== LEAKAGE-SAFE HOLDOUT RESULTS ===")
    for key, value in metrics.items():
        print(f"{key}: {value}")
    print("\n=== CLASSIFICATION REPORT ===")
    print(report)
    print("\nNew evaluation reports saved in:", REPORT_DIR)
if __name__ == "__main__":
    main()
