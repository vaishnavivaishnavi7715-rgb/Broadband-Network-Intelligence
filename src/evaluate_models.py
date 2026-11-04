import json
from pathlib import Path
import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "broadband_candidate_telemetry.csv"
MODEL_DIR = ROOT / "models"
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
MODEL_FILES = [
    "broadband_anomaly_candidate.joblib",
    "broadband_anomaly_candidate_v2.joblib",
    "broadband_anomaly_model.joblib",
]
def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    required = FEATURES + [TARGET]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing dataset columns: {missing}")
    df = df[required].dropna()
    X = df[FEATURES]
    y = df[TARGET].astype(int)
    if set(y.unique()) != {0, 1}:
        raise ValueError(
            f"Expected binary labels 0 and 1; found {sorted(y.unique())}"
        )
    results = []
    for filename in MODEL_FILES:
        model_path = MODEL_DIR / filename
        if not model_path.exists():
            print(f"Skipping missing model: {filename}")
            continue
        saved = joblib.load(model_path)
        model = saved["model"]
        features = saved.get("features", FEATURES)
        if list(features) != FEATURES:
            raise ValueError(
                f"Feature mismatch in {filename}: {features}"
            )
        predictions = model.predict(X).astype(int)
        probabilities = None
        if hasattr(model, "predict_proba"):
            classes = list(model.classes_)
            if 1 in classes:
                probabilities = model.predict_proba(X)[:, classes.index(1)]
        matrix = confusion_matrix(y, predictions, labels=[0, 1])
        metrics = {
            "model": filename,
            "rows_evaluated": int(len(y)),
            "accuracy": float(accuracy_score(y, predictions)),
            "precision": float(
                precision_score(y, predictions, zero_division=0)
            ),
            "recall": float(
                recall_score(y, predictions, zero_division=0)
            ),
            "f1_score": float(
                f1_score(y, predictions, zero_division=0)
            ),
            "confusion_matrix": {
                "true_normal_predicted_normal": int(matrix[0, 0]),
                "true_normal_predicted_anomaly": int(matrix[0, 1]),
                "true_anomaly_predicted_normal": int(matrix[1, 0]),
                "true_anomaly_predicted_anomaly": int(matrix[1, 1]),
            },
            "classification_report": classification_report(
                y,
                predictions,
                labels=[0, 1],
                target_names=["normal", "anomaly"],
                output_dict=True,
                zero_division=0,
            ),
        }
        if probabilities is not None:
            metrics["roc_auc"] = float(roc_auc_score(y, probabilities))
        results.append(metrics)
        print(f"\n{'=' * 60}")
        print(f"MODEL: {filename}")
        print(f"Rows evaluated: {len(y)}")
        for key in ["accuracy", "precision", "recall", "f1_score", "roc_auc"]:
            if key in metrics:
                print(f"{key}: {metrics[key]:.4f}")
        print("Confusion matrix (rows=true, columns=predicted; [normal, anomaly]):")
        print(matrix)
        print("\nClassification report:")
        print(
            classification_report(
                y,
                predictions,
                labels=[0, 1],
                target_names=["normal", "anomaly"],
                zero_division=0,
            )
        )
    if not results:
        raise RuntimeError("No models were evaluated.")
    summary = pd.DataFrame([
        {
            key: result[key]
            for key in [
                "model", "rows_evaluated", "accuracy",
                "precision", "recall", "f1_score", "roc_auc"
            ]
            if key in result
        }
        for result in results
    ])
    summary_path = REPORT_DIR / "anomaly_model_comparison.csv"
    json_path = REPORT_DIR / "anomaly_model_evaluation.json"
    summary.to_csv(summary_path, index=False)
    with json_path.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)
    print("\nEvaluation finished.")
    print(f"CSV summary: {summary_path}")
    print(f"Detailed JSON report: {json_path}")
    print("\nModel comparison:")
    print(summary.to_string(index=False))
    print(
        "\nIMPORTANT: These scores are valid as independent test scores only "
        "if these rows were not used to train or tune the evaluated models."
    )
if __name__ == "__main__":
    main()
