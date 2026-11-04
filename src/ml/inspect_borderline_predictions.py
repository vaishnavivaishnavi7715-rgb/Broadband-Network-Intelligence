
from pathlib import Path

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "broadband_anomaly_model.joblib"

scenarios = [
    {
        "scenario": "Healthy",
        "latency_ms": 30,
        "download_mbps": 200,
        "upload_mbps": 30,
        "packet_loss_pct": 0.2,
        "signal_strength_dbm": -55,
        "connection_drops": 0,
    },
    {
        "scenario": "Slightly elevated latency",
        "latency_ms": 75,
        "download_mbps": 150,
        "upload_mbps": 25,
        "packet_loss_pct": 0.5,
        "signal_strength_dbm": -65,
        "connection_drops": 0,
    },
    {
        "scenario": "Moderate packet loss",
        "latency_ms": 60,
        "download_mbps": 120,
        "upload_mbps": 20,
        "packet_loss_pct": 2.5,
        "signal_strength_dbm": -65,
        "connection_drops": 1,
    },
    {
        "scenario": "Weak signal, good speed",
        "latency_ms": 45,
        "download_mbps": 100,
        "upload_mbps": 20,
        "packet_loss_pct": 0.5,
        "signal_strength_dbm": -82,
        "connection_drops": 1,
    },
]

saved = joblib.load(MODEL_PATH)
model = saved["model"]
features = saved["features"]

data = pd.DataFrame(scenarios)
X = data[features]

classes = list(model.classes_)
normal_index = classes.index(0)
anomaly_index = classes.index(1)
probabilities = model.predict_proba(X)

for i, scenario in enumerate(scenarios):
    print(f"\n{scenario['scenario']}")
    print("-" * 40)
    print(f"Normal probability:  {probabilities[i, normal_index]:.2%}")
    print(f"Anomaly probability: {probabilities[i, anomaly_index]:.2%}")
    print("Measurements:")
    for feature in features:
        print(f"  {feature}: {scenario[feature]}")

print("\nRead-only inspection: no model or API files changed.")
