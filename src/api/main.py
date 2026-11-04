
from pathlib import Path
import sqlite3

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field


PROJECT_ROOT = Path(__file__).resolve().parents[2]

ANOMALY_MODEL_PATH = (
    PROJECT_ROOT / "models" / "broadband_anomaly_model.joblib"
)
ISSUE_MODEL_PATH = (
    PROJECT_ROOT / "models" / "network_issue_classifier.joblib"
)
HISTORY_DB_PATH = PROJECT_ROOT / "data" / "prediction_history.db"


# Load the existing trained models. Do not retrain or replace them.
anomaly_saved_model = joblib.load(ANOMALY_MODEL_PATH)
anomaly_model = anomaly_saved_model["model"]
ANOMALY_FEATURES = anomaly_saved_model["features"]

issue_saved_model = joblib.load(ISSUE_MODEL_PATH)
issue_model = issue_saved_model["model"]
ISSUE_FEATURES = issue_saved_model["features"]


app = FastAPI(
    title="Broadband Network Intelligence API",
    description=(
        "Experimental broadband anomaly detection and network issue "
        "classification using simulated telemetry."
    ),
    version="1.2.0",
)


class NetworkMetrics(BaseModel):
    latency_ms: float = Field(ge=0, le=10000)
    download_mbps: float = Field(ge=0, le=100000)
    upload_mbps: float = Field(ge=0, le=100000)
    packet_loss_pct: float = Field(ge=0, le=100)
    signal_strength_dbm: float = Field(ge=-150, le=0)
    connection_drops: int = Field(ge=0, le=100000)


@app.get("/")
def home():
    return {
        "message": "Broadband Network Intelligence API is running",
        "docs": "/docs",
        "endpoints": ["/health", "/predict"],
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": (
            anomaly_model is not None and issue_model is not None
        ),
        "anomaly_model_loaded": anomaly_model is not None,
        "issue_model_loaded": issue_model is not None,
    }


def explain_network_metrics(values):
    explanations = []
    recommendations = []

    if values["latency_ms"] >= 100:
        explanations.append({
            "metric": "latency_ms",
            "value": values["latency_ms"],
            "observation": "High latency",
        })
        recommendations.append(
            "Check network congestion, routing, and response times."
        )
    elif values["latency_ms"] >= 60:
        explanations.append({
            "metric": "latency_ms",
            "value": values["latency_ms"],
            "observation": "Elevated latency",
        })
        recommendations.append(
            "Monitor latency over time and check for congestion."
        )

    if values["packet_loss_pct"] >= 3:
        explanations.append({
            "metric": "packet_loss_pct",
            "value": values["packet_loss_pct"],
            "observation": "High packet loss",
        })
        recommendations.append(
            "Inspect link quality, interface errors, and congestion."
        )
    elif values["packet_loss_pct"] >= 1:
        explanations.append({
            "metric": "packet_loss_pct",
            "value": values["packet_loss_pct"],
            "observation": "Elevated packet loss",
        })
        recommendations.append(
            "Repeat packet-loss measurements and inspect link stability."
        )

    if values["signal_strength_dbm"] <= -80:
        explanations.append({
            "metric": "signal_strength_dbm",
            "value": values["signal_strength_dbm"],
            "observation": "Weak signal",
        })
        recommendations.append(
            "Check modem placement, interference, cabling, and signal levels."
        )

    if values["download_mbps"] < 25:
        explanations.append({
            "metric": "download_mbps",
            "value": values["download_mbps"],
            "observation": "Low download throughput",
        })
        recommendations.append(
            "Compare measured throughput with the subscribed plan and "
            "check for congestion or device limitations."
        )

    if values["upload_mbps"] < 5:
        explanations.append({
            "metric": "upload_mbps",
            "value": values["upload_mbps"],
            "observation": "Low upload throughput",
        })
        recommendations.append(
            "Run another upload test and inspect upstream link conditions."
        )

    if values["connection_drops"] >= 4:
        explanations.append({
            "metric": "connection_drops",
            "value": values["connection_drops"],
            "observation": "Frequent connection drops",
        })
        recommendations.append(
            "Review connection logs, modem stability, and physical links."
        )

    if not explanations:
        explanations.append({
            "metric": None,
            "value": None,
            "observation": (
                "No configured metric warning thresholds were triggered."
            ),
        })
        recommendations.append(
            "Continue monitoring; absence of warnings does not guarantee "
            "that the network is healthy."
        )

    return explanations, list(dict.fromkeys(recommendations))


def save_prediction(values, result):
    """Save a prediction to the existing SQLite history table."""
    with sqlite3.connect(HISTORY_DB_PATH, timeout=10) as conn:
        conn.execute(
            """
            INSERT INTO prediction_history (
                timestamp,
                latency_ms,
                download_mbps,
                upload_mbps,
                packet_loss_pct,
                signal_strength_dbm,
                connection_drops,
                prediction,
                is_anomaly,
                anomaly_probability,
                predicted_issue,
                issue_confidence
            )
            VALUES (
                datetime('now'),
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
            """,
            (
                values["latency_ms"],
                values["download_mbps"],
                values["upload_mbps"],
                values["packet_loss_pct"],
                values["signal_strength_dbm"],
                values["connection_drops"],
                result["prediction"],
                int(result["is_anomaly"]),
                result["anomaly_probability"],
                result["predicted_issue"],
                result["issue_confidence"],
            ),
        )


@app.post("/predict")
def predict(metrics: NetworkMetrics):
    values = metrics.model_dump()

    anomaly_input = pd.DataFrame(
        [values],
        columns=ANOMALY_FEATURES,
    )

    anomaly_prediction = int(anomaly_model.predict(anomaly_input)[0])

    anomaly_probabilities = anomaly_model.predict_proba(
        anomaly_input
    )[0]
    anomaly_classes = list(anomaly_model.classes_)

    anomaly_probability = (
        float(anomaly_probabilities[anomaly_classes.index(1)])
        if 1 in anomaly_classes
        else 0.0
    )

    issue_input = pd.DataFrame(
        [values],
        columns=ISSUE_FEATURES,
    )

    predicted_issue = str(issue_model.predict(issue_input)[0])

    issue_probabilities = issue_model.predict_proba(issue_input)[0]
    issue_classes = list(issue_model.classes_)

    issue_confidence = float(
        issue_probabilities[issue_classes.index(predicted_issue)]
    )

    explanations, recommendations = explain_network_metrics(values)

    result = {
        "prediction": (
            "Anomaly" if anomaly_prediction == 1 else "Normal"
        ),
        "is_anomaly": bool(anomaly_prediction),
        "anomaly_probability": round(anomaly_probability, 4),
        "predicted_issue": predicted_issue,
        "issue_confidence": round(issue_confidence, 4),
        "explanations": explanations,
        "recommendations": recommendations,
        "input_metrics": values,
        "note": (
            "Experimental models trained on simulated data with "
            "synthetic rule-generated issue labels. Explanations use "
            "fixed metric thresholds and do not establish causation. "
            "Predictions and recommendations are not validated "
            "diagnoses for real-world networks."
        ),
    }

    # Persist the prediction before returning the successful response.
    save_prediction(values, result)

    return result
