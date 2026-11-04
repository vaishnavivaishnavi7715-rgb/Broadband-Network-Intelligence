
from pathlib import Path
import json

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from src.ml.generate_data import generate_broadband_data


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "broadband_anomaly_model.joblib"
REPORT_PATH = PROJECT_ROOT / "reports" / "fresh_data_evaluation.json"


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    # Generate a fresh dataset with a different random seed.
    fresh_df = generate_broadband_data(rows=2000, seed=2026)

    # Load the existing model.
    saved = joblib.load(MODEL_PATH)
    model = saved["model"]
    features = saved["features"]

    X_fresh = fresh_df[features]
    y_true = fresh_df["is_anomaly"].astype(int)
    y_pred = model.predict(X_fresh)

    results = {
        "evaluation_type": "fresh synthetic dataset",
        "test_samples": int(len(fresh_df)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(
            precision_score(y_true, y_pred, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, y_pred, zero_division=0)
        ),
        "f1_score": float(
            f1_score(y_true, y_pred, zero_division=0)
        ),
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=[0, 1]
        ).tolist(),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=[0, 1],
            target_names=["Normal", "Anomaly"],
            output_dict=True,
            zero_division=0,
        ),
        "important_note": (
            "This is a fresh synthetic dataset, not real network data. "
            "The generator uses the same fixed random seed each time, "
            "so this checks a newly generated dataset but is not an "
            "independent random-seed evaluation."
        ),
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )

    print("\n=== Fresh Synthetic Data Evaluation ===")
    print(f"Test samples: {results['test_samples']}")
    print(f"Accuracy:     {results['accuracy']:.4f}")
    print(f"Precision:    {results['precision']:.4f}")
    print(f"Recall:       {results['recall']:.4f}")
    print(f"F1-score:     {results['f1_score']:.4f}")

    print("\nConfusion matrix (rows=true, columns=predicted):")
    print(results["confusion_matrix"])
    print(f"\nReport saved to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
