
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
    roc_auc_score,
    classification_report,
)

# Project paths
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = BASE_DIR / "data" / "broadband_telemetry.csv"
MODEL_PATH = BASE_DIR / "models" / "broadband_anomaly_model.joblib"
REPORT_PATH = BASE_DIR / "reports" / "model_evaluation.json"

TARGET = "is_anomaly"


def main():
    # Check required files
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    # Load data and saved model package
    df = pd.read_csv(DATA_PATH)
    saved_data = joblib.load(MODEL_PATH)

    if not isinstance(saved_data, dict):
        raise TypeError("Expected the saved model file to contain a dictionary.")

    if "model" not in saved_data or "features" not in saved_data:
        raise KeyError(
            "Saved model must contain both 'model' and 'features'."
        )

    model = saved_data["model"]
    features = saved_data["features"]

    if TARGET not in df.columns:
        raise ValueError(
            f"Target column '{TARGET}' not found. "
            f"Available columns: {list(df.columns)}"
        )

    missing_features = [col for col in features if col not in df.columns]
    if missing_features:
        raise ValueError(f"Dataset is missing features: {missing_features}")

    X = df[features]
    y = df[TARGET].astype(int)

    # Recreate the held-out split used by training.
    # These settings must match train_model.py.
    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # Generate predictions
    predictions = model.predict(X_test)

    if not hasattr(model, "predict_proba"):
        raise AttributeError(
            "The saved model does not support predict_proba()."
        )

    probabilities = model.predict_proba(X_test)[:, 1]

    # Calculate metrics
    tn, fp, fn, tp = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1],
    ).ravel()

    report = {
        "dataset": DATA_PATH.name,
        "evaluation_type": "held-out synthetic test split",
        "test_samples": int(len(y_test)),
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
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
        },
        "classification_report": classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["Normal", "Anomaly"],
            output_dict=True,
            zero_division=0,
        ),
        "important_note": (
            "Evaluation uses synthetic data. Results do not establish "
            "performance on real broadband networks."
        ),
    }

    # Save report
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    # Display metrics
    print("\n=== Broadband Anomaly Model Evaluation ===")
    print(f"Test samples: {report['test_samples']}")
    print(f"Accuracy:     {report['accuracy']:.4f}")
    print(f"Precision:    {report['precision']:.4f}")
    print(f"Recall:       {report['recall']:.4f}")
    print(f"F1-score:     {report['f1_score']:.4f}")
    print(f"ROC-AUC:      {report['roc_auc']:.4f}")

    print("\n=== Confusion Matrix ===")
    print(f"True negatives:  {tn}")
    print(f"False positives: {fp}")
    print(f"False negatives: {fn}")
    print(f"True positives:  {tp}")

    print("\n=== Classification Report ===")
    print(
        classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["Normal", "Anomaly"],
            zero_division=0,
        )
    )

    print(f"Full JSON report saved to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
