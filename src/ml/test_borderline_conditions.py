
from pathlib import Path

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "broadband_anomaly_model.joblib"

# Expected labels below are test assumptions, not verified real-world truth.
scenarios = [
    {
        "scenario": "Healthy connection",
        "latency_ms": 30,
        "download_mbps": 200,
        "upload_mbps": 30,
        "packet_loss_pct": 0.2,
        "signal_strength_dbm": -55,
        "connection_drops": 0,
        "expected_anomaly": 0,
    },
    {
        "scenario": "Slightly elevated latency",
        "latency_ms": 75,
        "download_mbps": 150,
        "upload_mbps": 25,
        "packet_loss_pct": 0.5,
        "signal_strength_dbm": -65,
        "connection_drops": 0,
        "expected_anomaly": 0,
    },
    {
        "scenario": "Moderate packet loss",
        "latency_ms": 60,
        "download_mbps": 120,
        "upload_mbps": 20,
        "packet_loss_pct": 2.5,
        "signal_strength_dbm": -65,
        "connection_drops": 1,
        "expected_anomaly": 0,
    },
    {
        "scenario": "Weak signal, good speed",
        "latency_ms": 45,
        "download_mbps": 100,
        "upload_mbps": 20,
        "packet_loss_pct": 0.5,
        "signal_strength_dbm": -82,
        "connection_drops": 1,
        "expected_anomaly": 0,
    },
    {
        "scenario": "High latency and packet loss",
        "latency_ms": 140,
        "download_mbps": 80,
        "upload_mbps": 15,
        "packet_loss_pct": 4.0,
        "signal_strength_dbm": -70,
        "connection_drops": 3,
        "expected_anomaly": 1,
    },
    {
        "scenario": "Severe weak signal",
        "latency_ms": 100,
        "download_mbps": 20,
        "upload_mbps": 5,
        "packet_loss_pct": 2.0,
        "signal_strength_dbm": -92,
        "connection_drops": 3,
        "expected_anomaly": 1,
    },
]

def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    saved = joblib.load(MODEL_PATH)
    model = saved["model"]
    features = saved["features"]

    data = pd.DataFrame(scenarios)
    X = data[features]
    y_true = data["expected_anomaly"].astype(int)

    classes = list(model.classes_)
    if 1 not in classes:
        raise ValueError("Model does not contain anomaly class 1.")

    probabilities = model.predict_proba(X)[:, classes.index(1)]

    print("\n=== Borderline Network Conditions ===")
    print("Expected labels are illustrative test assumptions.")
    print("\nScenario                         Expected  Probability")

    for name, expected, probability in zip(
        data["scenario"], y_true, probabilities
    ):
        print(f"{name:<32} {expected:^8} {probability:>10.2%}")

    print("\n=== Threshold Comparison ===")
    print("Threshold | Agreement | Missed anomalies | False alarms")

    for threshold in (0.50, 0.60, 0.70):
        predicted = (probabilities >= threshold).astype(int)
        agreement = (predicted == y_true).mean()
        missed = ((y_true == 1) & (predicted == 0)).sum()
        false_alarms = ((y_true == 0) & (predicted == 1)).sum()

        print(
            f"{threshold:>8.0%} | {agreement:>8.2%} | "
            f"{missed:>16} | {false_alarms:>12}"
        )

    print("\nNo model or API files were changed.")

if __name__ == "__main__":
    main()
