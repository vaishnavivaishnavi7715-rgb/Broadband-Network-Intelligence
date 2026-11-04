import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

MODEL_PATH = "models/broadband_anomaly_model.joblib"
OUTPUT_PATH = "reports/overlapping_issues_test.csv"

scenarios = [
    {
        "scenario": "Healthy connection",
        "latency_ms": 25,
        "download_mbps": 250,
        "upload_mbps": 40,
        "packet_loss_pct": 0.1,
        "signal_strength_dbm": -50,
        "connection_drops": 0,
        "expected_anomaly": 0,
    },
    {
        "scenario": "Weak signal + high latency",
        "latency_ms": 180,
        "download_mbps": 35,
        "upload_mbps": 8,
        "packet_loss_pct": 1.5,
        "signal_strength_dbm": -92,
        "connection_drops": 2,
        "expected_anomaly": 1,
    },
    {
        "scenario": "Packet loss + connection drops",
        "latency_ms": 90,
        "download_mbps": 60,
        "upload_mbps": 12,
        "packet_loss_pct": 8.0,
        "signal_strength_dbm": -65,
        "connection_drops": 8,
        "expected_anomaly": 1,
    },
    {
        "scenario": "Low bandwidth + high latency",
        "latency_ms": 240,
        "download_mbps": 12,
        "upload_mbps": 3,
        "packet_loss_pct": 1.0,
        "signal_strength_dbm": -60,
        "connection_drops": 1,
        "expected_anomaly": 1,
    },
    {
        "scenario": "Multiple simultaneous issues",
        "latency_ms": 350,
        "download_mbps": 8,
        "upload_mbps": 2,
        "packet_loss_pct": 12.0,
        "signal_strength_dbm": -95,
        "connection_drops": 12,
        "expected_anomaly": 1,
    },
    {
        "scenario": "Moderate degradation",
        "latency_ms": 65,
        "download_mbps": 120,
        "upload_mbps": 25,
        "packet_loss_pct": 1.0,
        "signal_strength_dbm": -72,
        "connection_drops": 1,
        "expected_anomaly": 0,
    },
    {
        "scenario": "Weak signal + packet loss",
        "latency_ms": 110,
        "download_mbps": 45,
        "upload_mbps": 10,
        "packet_loss_pct": 5.0,
        "signal_strength_dbm": -90,
        "connection_drops": 5,
        "expected_anomaly": 1,
    },
]


def main():
    model_data = joblib.load(MODEL_PATH)
    model = model_data["model"]
    features = model_data["features"]

    data = pd.DataFrame(scenarios)
    predictions = model.predict(data[features])

    data["predicted_anomaly"] = predictions
    data["anomaly_probability_pct"] = float("nan")

    if hasattr(model, "predict_proba"):
        anomaly_class_index = list(model.classes_).index(1)
        data["anomaly_probability_pct"] = (
            model.predict_proba(data[features])[:, anomaly_class_index]
            * 100
        ).round(2)

    data["prediction_correct_vs_expectation"] = (
        data["expected_anomaly"] == data["predicted_anomaly"]
    )

    print("\n=== Overlapping Network Issues Stress Test ===")
    print(
        data[
            [
                "scenario",
                "expected_anomaly",
                "predicted_anomaly",
                "anomaly_probability_pct",
                "prediction_correct_vs_expectation",
            ]
        ].to_string(index=False)
    )

    accuracy = accuracy_score(
        data["expected_anomaly"],
        data["predicted_anomaly"],
    )
    print(f"\nAgreement with expected labels: {accuracy:.2%}")

    print("\nClassification report:")
    print(
        classification_report(
            data["expected_anomaly"],
            data["predicted_anomaly"],
            labels=[0, 1],
            target_names=["Expected normal", "Expected anomaly"],
            zero_division=0,
        )
    )

    data.to_csv(OUTPUT_PATH, index=False)
    print(f"\nResults saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()