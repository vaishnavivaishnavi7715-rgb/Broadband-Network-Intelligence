from pathlib import Path
import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import train_test_split
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "broadband_candidate_telemetry.csv"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
MODEL_FILES = {
    "Original": "broadband_anomaly_model.joblib",
    "Candidate V1": "broadband_anomaly_candidate.joblib",
    "Candidate V2": "broadband_anomaly_candidate_v2.joblib",
}
def main():
    df = pd.read_csv(DATA)
    features = [
        "latency_ms",
        "download_mbps",
        "upload_mbps",
        "packet_loss_pct",
        "signal_strength_dbm",
        "connection_drops",
    ]
    X = df[features]
    y = df["is_anomaly"]
    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )
    rows = []
    for name, filename in MODEL_FILES.items():
        saved = joblib.load(MODELS / filename)
        model = saved["model"]
        model_features = saved["features"]
        predictions = model.predict(X_test[model_features])
        metrics = {
            "model": name,
            "accuracy": accuracy_score(y_test, predictions),
            "precision": precision_score(
                y_test, predictions, zero_division=0
            ),
            "recall": recall_score(
                y_test, predictions, zero_division=0
            ),
            "f1_score": f1_score(
                y_test, predictions, zero_division=0
            ),
        }
        rows.append(metrics)
        print(f"\n{'=' * 55}")
        print(name)
        print("=" * 55)
        print(classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["Normal", "Anomaly"],
            zero_division=0,
        ))
        print("Confusion matrix [[TN, FP], [FN, TP]]:")
        print(confusion_matrix(y_test, predictions, labels=[0, 1]))
    results = pd.DataFrame(rows)
    REPORTS.mkdir(parents=True, exist_ok=True)
    output = REPORTS / "three_model_evaluation.csv"
    results.to_csv(output, index=False)
    print("\nSUMMARY")
    print(results.round(4).to_string(index=False))
    print(f"\nResults saved to: {output}")
    print("\nSynthetic-data evaluation only; no production model was changed.")
if __name__ == "__main__":
    main()
