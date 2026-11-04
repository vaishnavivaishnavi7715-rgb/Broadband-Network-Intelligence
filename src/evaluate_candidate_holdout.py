from pathlib import Path
import json
import joblib
import pandas as pd
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
from sklearn.model_selection import train_test_split
ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "broadband_candidate_telemetry.csv"
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
    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=14,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    matrix = confusion_matrix(y_test, predictions, labels=[0, 1])
    metrics = {
        "evaluation_type": "Fresh model, stratified 80/20 holdout",
        "dataset": DATA_PATH.name,
        "total_rows": int(len(df)),
        "training_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(
            precision_score(y_test, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(y_test, predictions, zero_division=0)
        ),
        "f1_score": float(
            f1_score(y_test, predictions, zero_division=0)
        ),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "confusion_matrix": matrix.tolist(),
        "warning": (
            "Synthetic data. Holdout reliability assumes independent "
            "records and no duplicate or related records across the split."
        ),
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    with (REPORT_DIR / "candidate_holdout_metrics.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(metrics, file, indent=2)
    report = classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["Normal", "Anomaly"],
        zero_division=0,
    )
    with (REPORT_DIR / "candidate_holdout_report.txt").open(
        "w", encoding="utf-8"
    ) as file:
        file.write(json.dumps(metrics, indent=2))
        file.write("\n\nClassification report:\n")
        file.write(report)
    pd.DataFrame(
        matrix,
        index=["Actual Normal", "Actual Anomaly"],
        columns=["Predicted Normal", "Predicted Anomaly"],
    ).to_csv(REPORT_DIR / "candidate_holdout_confusion_matrix.csv")
    print("\n=== HELD-OUT EVALUATION ===")
    for key, value in metrics.items():
        print(f"{key}: {value}")
    print("\n=== CLASSIFICATION REPORT ===")
    print(report)
    print(f"\nReports saved to: {REPORT_DIR}")
if __name__ == "__main__":
    main()
