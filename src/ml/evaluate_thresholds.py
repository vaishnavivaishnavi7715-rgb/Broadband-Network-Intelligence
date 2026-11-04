
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from src.ml.generate_data import generate_broadband_data


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "broadband_anomaly_model.joblib"


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    # Generate synthetic data using a different seed.
    fresh_df = generate_broadband_data(rows=5000, seed=31415)

    saved = joblib.load(MODEL_PATH)
    model = saved["model"]
    features = saved["features"]

    X = fresh_df[features]
    y_true = fresh_df["is_anomaly"].astype(int)

    classes = list(model.classes_)
    if 1 not in classes:
        raise ValueError("Model does not contain anomaly class 1.")

    probabilities = model.predict_proba(X)[:, classes.index(1)]

    print("\n=== Threshold Evaluation: Fresh Synthetic Data ===")
    print(f"Test samples: {len(fresh_df)}")
    print("Threshold | Accuracy | Precision | Recall | F1 | Missed | False alarms")

    for threshold in (0.50, 0.60, 0.70):
        predictions = (probabilities >= threshold).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            y_true, predictions, labels=[0, 1]
        ).ravel()

        print(
            f"{threshold:>8.0%} | "
            f"{accuracy_score(y_true, predictions):>8.4f} | "
            f"{precision_score(y_true, predictions, zero_division=0):>9.4f} | "
            f"{recall_score(y_true, predictions, zero_division=0):>6.4f} | "
            f"{f1_score(y_true, predictions, zero_division=0):>4.4f} | "
            f"{fn:>6} | {fp:>12}"
        )

    print(
        "\nNote: This is synthetic data from the same generator, "
        "not validation on real broadband telemetry."
    )
    print("No model or API files were changed.")


if __name__ == "__main__":
    main()
