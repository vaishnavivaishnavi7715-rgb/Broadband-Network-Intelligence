from pathlib import Path
import joblib
import pandas as pd
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"
FEATURES = [
    "latency_ms",
    "download_mbps",
    "upload_mbps",
    "packet_loss_pct",
    "signal_strength_dbm",
    "connection_drops",
]
SCENARIOS = [
    {"name": "Healthy", "latency_ms": 30, "download_mbps": 200, "upload_mbps": 30, "packet_loss_pct": 0.2, "signal_strength_dbm": -55, "connection_drops": 0},
    {"name": "Slightly elevated latency", "latency_ms": 75, "download_mbps": 150, "upload_mbps": 25, "packet_loss_pct": 0.5, "signal_strength_dbm": -65, "connection_drops": 0},
    {"name": "Moderate packet loss", "latency_ms": 60, "download_mbps": 120, "upload_mbps": 20, "packet_loss_pct": 2.5, "signal_strength_dbm": -65, "connection_drops": 1},
    {"name": "Weak signal, good speed", "latency_ms": 45, "download_mbps": 100, "upload_mbps": 20, "packet_loss_pct": 0.5, "signal_strength_dbm": -82, "connection_drops": 1},
    {"name": "Severe weak signal", "latency_ms": 100, "download_mbps": 20, "upload_mbps": 5, "packet_loss_pct": 2.0, "signal_strength_dbm": -92, "connection_drops": 3},
    {"name": "Severe packet loss", "latency_ms": 90, "download_mbps": 60, "upload_mbps": 12, "packet_loss_pct": 8.0, "signal_strength_dbm": -65, "connection_drops": 8},
    {"name": "Severe high latency", "latency_ms": 220, "download_mbps": 200, "upload_mbps": 30, "packet_loss_pct": 0.5, "signal_strength_dbm": -55, "connection_drops": 0},
]
MODEL_FILES = {
    "Original": "broadband_anomaly_model.joblib",
    "Candidate V1": "broadband_anomaly_candidate.joblib",
    "Candidate V2": "broadband_anomaly_candidate_v2.joblib",
}
def load_model(filename):
    saved = joblib.load(MODELS_DIR / filename)
    return saved["model"], saved["features"]
def main():
    models = {
        name: load_model(filename)
        for name, filename in MODEL_FILES.items()
    }
    rows = []
    for scenario in SCENARIOS:
        inputs = pd.DataFrame(
            [{feature: scenario[feature] for feature in FEATURES}],
            columns=FEATURES,
        )
        row = {"scenario": scenario["name"]}
        for name, (model, model_features) in models.items():
            probabilities = model.predict_proba(inputs[model_features])[0]
            anomaly_index = list(model.classes_).index(1)
            row[f"{name} anomaly %"] = round(
                probabilities[anomaly_index] * 100, 2
            )
        rows.append(row)
    results = pd.DataFrame(rows)
    print("\nTHREE-MODEL SCENARIO COMPARISON")
    print("=" * 95)
    print(results.to_string(index=False))
    output = PROJECT_ROOT / "reports" / "candidate_comparison_all.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output, index=False)
    print(f"\nReport saved to: {output}")
    print("Original model and API were not changed.")
if __name__ == "__main__":
    main()
