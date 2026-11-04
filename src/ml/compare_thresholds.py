
from pathlib import Path

import joblib
import pandas as pd

from src.ml.test_overlapping_issues import scenarios


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "broadband_anomaly_model.joblib"

saved = joblib.load(MODEL_PATH)
model = saved["model"]
features = saved["features"]

data = pd.DataFrame(scenarios)
X = data[features]
y_true = data["expected_anomaly"].astype(int)

classes = list(model.classes_)
if 1 not in classes:
    raise ValueError("The model does not contain anomaly class 1.")

probabilities = model.predict_proba(X)[:, classes.index(1)]

print("\n=== Threshold Comparison ===")
print("Threshold | Accuracy | Missed anomalies | False alarms")

for threshold in (0.50, 0.60, 0.70):
    predictions = (probabilities >= threshold).astype(int)

    accuracy = (predictions == y_true).mean()
    missed = ((y_true == 1) & (predictions == 0)).sum()
    false_alarms = ((y_true == 0) & (predictions == 1)).sum()

    print(
        f"{threshold:>8.0%} | "
        f"{accuracy:>7.2%} | "
        f"{missed:>16} | "
        f"{false_alarms:>12}"
    )

print("\n=== Individual Scenario Probabilities ===")
for scenario, expected, probability in zip(
    data["scenario"], y_true, probabilities
):
    print(
        f"{scenario}: expected={expected}, "
        f"probability={probability:.2%}"
    )

print("\nComparison only. No model or API files were changed.")
