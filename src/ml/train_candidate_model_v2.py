from pathlib import Path
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "broadband_candidate_telemetry.csv"
OUTPUT = ROOT / "models" / "broadband_anomaly_candidate_v2.joblib"
FEATURES = [
    "latency_ms",
    "download_mbps",
    "upload_mbps",
    "packet_loss_pct",
    "signal_strength_dbm",
    "connection_drops",
]
def main():
    df = pd.read_csv(DATA)
    X, y = df[FEATURES], df["is_anomaly"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=14,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print("CANDIDATE V2 EVALUATION")
    print(classification_report(
        y_test, predictions,
        target_names=["Normal", "Anomaly"],
        zero_division=0,
    ))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, predictions))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "features": FEATURES}, OUTPUT)
    print(f"\nCandidate saved to: {OUTPUT}")
    print("Original model and API remain unchanged.")
if __name__ == "__main__":
    main()
