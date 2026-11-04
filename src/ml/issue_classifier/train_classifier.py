
from pathlib import Path
import json

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


BASE_DIR = Path(__file__).resolve().parents[3]

DATA_PATH = BASE_DIR / "data" / "broadband_telemetry.csv"
MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports"

MODEL_PATH = MODEL_DIR / "network_issue_classifier.joblib"
REPORT_PATH = REPORT_DIR / "issue_classifier_evaluation.json"

FEATURES = [
    "latency_ms",
    "download_mbps",
    "upload_mbps",
    "packet_loss_pct",
    "signal_strength_dbm",
    "connection_drops",
]


def assign_issue(row):
    """Generate demonstration labels from telemetry rules."""

    if row["signal_strength_dbm"] <= -85:
        return "weak_signal"

    if row["packet_loss_pct"] >= 3:
        return "packet_loss"

    if row["connection_drops"] >= 4:
        return "unstable_connection"

    if row["latency_ms"] >= 100:
        return "high_latency"

    if row["download_mbps"] < 25 or row["upload_mbps"] < 5:
        return "low_bandwidth"

    return "normal"


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    missing = [column for column in FEATURES if column not in df.columns]
    if missing:
        raise ValueError(f"Missing dataset columns: {missing}")

    # These labels are generated for demonstration, not real diagnoses.
    df["network_issue"] = df.apply(assign_issue, axis=1)

    X = df[FEATURES]
    y = df["network_issue"]

    print("\n=== Network Issue Classifier ===")
    print(f"Total records: {len(df)}")
    print("\nClass counts:")
    print(y.value_counts().to_string())

    if y.nunique() < 2:
        raise ValueError("At least two issue classes are required.")

    class_counts = y.value_counts()
    use_stratify = (
        class_counts.min() >= 2
        and int(len(df) * 0.2) >= len(class_counts)
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y if use_stratify else None,
    )

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", RandomForestClassifier(
            n_estimators=150,
            random_state=42,
            class_weight="balanced",
        )),
    ])

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    report = classification_report(
        y_test,
        predictions,
        output_dict=True,
        zero_division=0,
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        {
            "model": model,
            "features": FEATURES,
            "classes": list(model.classes_),
        },
        MODEL_PATH,
    )

    output = {
        "dataset_rows": int(len(df)),
        "training_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "label_source": (
            "Synthetic rule-based demonstration labels; "
            "not verified real-world diagnoses"
        ),
        "accuracy": float(accuracy),
        "classification_report": report,
        "class_counts": {
            str(label): int(count)
            for label, count in y.value_counts().items()
        },
    }

    REPORT_PATH.write_text(
        json.dumps(output, indent=2),
        encoding="utf-8",
    )

    print(f"\nTraining records: {len(X_train)}")
    print(f"Test records: {len(X_test)}")
    print(f"Accuracy: {accuracy:.4f}")

    print("\nClassification report:")
    print(classification_report(
        y_test,
        predictions,
        zero_division=0,
    ))

    print(f"\nModel saved to: {MODEL_PATH}")
    print(f"Report saved to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
