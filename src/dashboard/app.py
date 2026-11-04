from pathlib import Path
import os
import sqlite3
import pandas as pd
import requests
import streamlit as st


# ============================================================
# Paths and configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "broadband_telemetry.csv"
HISTORY_DB = PROJECT_ROOT / "data" / "prediction_history.db"

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000").rstrip("/")
PREDICT_URL = f"{API_URL}/predict"

st.set_page_config(
    page_title="Broadband Network Intelligence",
    page_icon="N",
    layout="wide",
)

st.title("Broadband Network Intelligence Platform")
st.caption("Network telemetry monitoring, AI predictions, and prediction history")


# ============================================================
# Database
# ============================================================

def initialize_history_db():
    HISTORY_DB.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(HISTORY_DB) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS prediction_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                latency_ms REAL,
                download_mbps REAL,
                upload_mbps REAL,
                packet_loss_pct REAL,
                signal_strength_dbm REAL,
                connection_drops INTEGER,
                prediction TEXT,
                is_anomaly INTEGER,
                anomaly_probability REAL,
                predicted_issue TEXT,
                issue_confidence REAL
            )
            """
        )


def load_prediction_history(limit=500):
    with sqlite3.connect(HISTORY_DB) as conn:
        return pd.read_sql_query(
            """
            SELECT *
            FROM prediction_history
            ORDER BY id DESC
            LIMIT ?
            """,
            conn,
            params=(limit,),
        )


# ============================================================
# Dataset
# ============================================================

@st.cache_data
def load_data(path):
    return pd.read_csv(path)


# ============================================================
# API health
# ============================================================

def check_api():
    try:
        response = requests.get(
            f"{API_URL}/health",
            timeout=3,
        )
        response.raise_for_status()
        return True
    except requests.RequestException:
        return False


# ============================================================
# Initialize
# ============================================================

initialize_history_db()

with st.sidebar:
    st.header("System Status")

    if check_api():
        st.success("Prediction API is connected")
    else:
        st.error("Prediction API is offline")
        st.caption(
            "Start the API in another PowerShell terminal "
            "before making predictions."
        )

    st.write("API:", API_URL)
    st.write("Database:", HISTORY_DB.name)


# ============================================================
# Dataset overview
# ============================================================

st.header("Network Telemetry Overview")
if not DATA_PATH.exists():
    st.error(f"Dataset not found: {DATA_PATH}")
    st.stop()

try:
    df = load_data(str(DATA_PATH))
except Exception as exc:
    st.error(f"Could not load dataset: {exc}")
    st.stop()

if df.empty:
    st.warning("The dataset is empty.")
    st.stop()

col1, col2, col3 = st.columns(3)

col1.metric("Total Records", f"{len(df):,}")
col2.metric("Total Features", len(df.columns))
col3.metric("Missing Values", int(df.isna().sum().sum()))

with st.expander("View dataset"):
    st.dataframe(df.head(100), hide_index=True, width="stretch")
    st.write("Dataset columns:", df.columns.tolist())

numeric_columns = df.select_dtypes(include="number").columns.tolist()

if numeric_columns:
    st.subheader("Telemetry Distribution")
    selected_column = st.selectbox(
        "Choose a numeric feature",
        numeric_columns,
    )
    st.bar_chart(df[selected_column].dropna())


# ============================================================
# Network prediction
# ============================================================

st.divider()
st.header("Predict Network Condition")
st.write(
    "Enter the current network measurements. "
    "The FastAPI backend will run the trained prediction models."
)

with st.form("network_prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        latency_ms = st.number_input(
            "Latency (ms)",
            min_value=0.0,
            value=50.0,
            step=1.0,
        )

        download_mbps = st.number_input(
            "Download speed (Mbps)",
            min_value=0.0,
            value=100.0,
            step=1.0,
        )

        upload_mbps = st.number_input(
            "Upload speed (Mbps)",
            min_value=0.0,
            value=20.0,
            step=1.0,
        )

    with col2:
        packet_loss_pct = st.number_input(
            "Packet loss (%)",
            min_value=0.0,
            max_value=100.0,
            value=0.5,
            step=0.1,
        )

        signal_strength_dbm = st.number_input(
            "Signal strength (dBm)",
            min_value=-150.0,
            max_value=0.0,
            value=-60.0,
            step=1.0,
        )

        connection_drops = st.number_input(
            "Connection drops",
            min_value=0,
            value=0,
            step=1,
        )

    submitted = st.form_submit_button(
        "Analyze Network",
        type="primary",
        use_container_width=True,
    )

if submitted:
    payload = {
        "latency_ms": latency_ms,
        "download_mbps": download_mbps,
        "upload_mbps": upload_mbps,
        "packet_loss_pct": packet_loss_pct,
        "signal_strength_dbm": signal_strength_dbm,
        "connection_drops": connection_drops,
    }

    try:
        with st.spinner("Analyzing network telemetry..."):
            response = requests.post(
                PREDICT_URL,
                json=payload,
                timeout=30,
            )

        if response.ok:
            result = response.json()

            st.success("Prediction completed successfully")

            result_col1, result_col2 = st.columns(2)

            with result_col1:
                st.subheader("Anomaly Detection")
                st.write(
                    "Prediction:",
                    result.get("prediction", "Not provided"),
                )
                st.write(
                    "Anomaly detected:",
                    result.get("is_anomaly", "Not provided"),
                )
                st.write(
                    "Anomaly probability:",
                    result.get("anomaly_probability", "Not provided"),
                )

            with result_col2:
                st.subheader("Issue Classification")
                st.write(
                    "Predicted issue:",
                    result.get("predicted_issue", "Not provided"),
                )
                st.write(
                    "Issue confidence:",
                    result.get("issue_confidence", "Not provided"),
                )

            with st.expander("View complete API response"):
                st.json(result)

            st.info(
                "If the API saves predictions to the database, "
                "this result will appear in Prediction History."
            )

        else:
            st.error(
                f"Prediction API returned HTTP {response.status_code}"
            )
            st.code(response.text)

    except requests.RequestException as exc:
        st.error(f"Could not connect to the prediction API: {exc}")


# ============================================================
# Prediction history
# ============================================================

from pathlib import Path
import json
import pandas as pd
import streamlit as st
PROJECT_ROOT = Path(__file__).resolve().parents[2]
METRICS_PATH = PROJECT_ROOT / "reports" / "candidate_holdout_metrics.json"
MATRIX_PATH = PROJECT_ROOT / "reports" / "candidate_holdout_confusion_matrix.csv"
st.divider()
st.header("Model Evaluation")
st.caption("Held-out evaluation of a Random Forest model on synthetic telemetry.")
if not METRICS_PATH.exists():
    st.warning("Evaluation metrics not found. Run src/evaluate_candidate_holdout.py first.")
else:
    try:
        with METRICS_PATH.open("r", encoding="utf-8") as file:
            metrics = json.load(file)
        if metrics.get("evaluation_type") != "Fresh model, stratified 80/20 holdout":
            st.warning("The evaluation type differs from the expected holdout evaluation.")
        metric_cols = st.columns(5)
        metric_cols[0].metric("Accuracy", f"{metrics['accuracy']:.2%}")
        metric_cols[1].metric("Precision", f"{metrics['precision']:.2%}")
        metric_cols[2].metric("Recall", f"{metrics['recall']:.2%}")
        metric_cols[3].metric("F1-score", f"{metrics['f1_score']:.2%}")
        metric_cols[4].metric("ROC-AUC", f"{metrics['roc_auc']:.2%}")
        detail_cols = st.columns(3)
        detail_cols[0].metric("Total records", f"{metrics['total_rows']:,}")
        detail_cols[1].metric("Training records", f"{metrics['training_rows']:,}")
        detail_cols[2].metric("Test records", f"{metrics['test_rows']:,}")
        st.subheader("Confusion Matrix")
        if MATRIX_PATH.exists():
            matrix_df = pd.read_csv(MATRIX_PATH, index_col=0)
            st.dataframe(matrix_df, use_container_width=True)
        else:
            st.warning("Confusion matrix CSV not found.")
        st.warning(metrics.get("warning", "Check dataset provenance before interpreting these results."))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        st.error(f"Could not load evaluation results: {exc}")


# ============================================================
# Network Analytics
# ============================================================

st.divider()
st.header("Network Health Analytics")
try:
    analytics_df = load_data(str(DATA_PATH))

    required_columns = [
        "latency_ms",
        "download_mbps",
        "upload_mbps",
        "packet_loss_pct",
        "signal_strength_dbm",
        "connection_drops",
    ]

    missing_columns = [
        column for column in required_columns
        if column not in analytics_df.columns
    ]

    if missing_columns:
        st.warning(
            "Analytics unavailable. Missing columns: "
            + ", ".join(missing_columns)
        )
    else:
        metric1, metric2, metric3 = st.columns(3)

        avg_latency = analytics_df["latency_ms"].mean()
        avg_download = analytics_df["download_mbps"].mean()
        avg_packet_loss = analytics_df["packet_loss_pct"].mean()

        metric1.metric(
            "Average Latency",
            f"{avg_latency:.2f} ms",
        )
        metric2.metric(
            "Average Download",
            f"{avg_download:.2f} Mbps",
        )
        metric3.metric(
            "Average Packet Loss",
            f"{avg_packet_loss:.2f}%",
        )

        st.subheader("Latency Distribution")
        st.line_chart(
            analytics_df["latency_ms"].reset_index(drop=True),
            x_label="Record",
            y_label="Latency (ms)",
        )

        st.subheader("Download and Upload Speeds")
        st.line_chart(
            analytics_df[
                ["download_mbps", "upload_mbps"]
            ].reset_index(drop=True),
            x_label="Record",
            y_label="Speed (Mbps)",
        )

        st.subheader("Packet Loss")
        st.line_chart(
            analytics_df["packet_loss_pct"].reset_index(drop=True),
            x_label="Record",
            y_label="Packet Loss (%)",
        )

        st.subheader("Connection Drops")
        st.bar_chart(
            analytics_df["connection_drops"].value_counts().sort_index(),
            x_label="Number of Drops",
            y_label="Number of Records",
        )

except Exception as exc:
    st.error(f"Could not load network analytics: {exc}")


st.divider()
st.header("Prediction History")
if st.button("Refresh History"):
    st.cache_data.clear()
    st.rerun()

try:
    history = load_prediction_history(limit=500)

    if history.empty:
        st.info(
            "No prediction history available yet. "
            "Submit a network prediction after starting the API."
        )
    else:
        if "timestamp" in history.columns:
            history["timestamp"] = pd.to_datetime(
                history["timestamp"],
                format="mixed",
                utc=True,
                errors="coerce",
            )

            history = history.dropna(subset=["timestamp"]).copy()

            history["timestamp"] = history["timestamp"].dt.strftime(
                "%Y-%m-%d %H:%M:%S UTC"
            )

        st.metric("Saved Predictions", len(history))

        st.dataframe(
            history,
            hide_index=True,
            width="stretch",
        )

        csv_data = history.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Download History as CSV",
            data=csv_data,
            file_name="broadband_prediction_history.csv",
            mime="text/csv",
        )

except Exception as exc:
    st.error(f"Could not load prediction history: {exc}")


# ============================================================
# Footer
# ============================================================

st.divider()
st.caption(
    "Broadband Network Intelligence Platform | "
    "AI/ML portfolio project using synthetic telemetry data"
)
