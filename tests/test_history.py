import sqlite3
from pathlib import Path


DB_PATH = Path(__file__).resolve().parents[1] / "data" / "prediction_history.db"


def test_prediction_history_database_exists():
    assert DB_PATH.exists(), f"Database not found: {DB_PATH}"


def test_prediction_history_table_exists():
    assert DB_PATH.exists()

    with sqlite3.connect(DB_PATH) as conn:
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()

    assert ("prediction_history",) in tables


def test_prediction_history_has_expected_columns():
    assert DB_PATH.exists()

    with sqlite3.connect(DB_PATH) as conn:
        columns = {
            row[1]
            for row in conn.execute(
                "PRAGMA table_info(prediction_history)"
            ).fetchall()
        }

    expected_columns = {
        "id",
        "timestamp",
        "latency_ms",
        "download_mbps",
        "upload_mbps",
        "packet_loss_pct",
        "signal_strength_dbm",
        "connection_drops",
        "prediction",
        "is_anomaly",
        "anomaly_probability",
        "predicted_issue",
        "issue_confidence",
    }

    assert expected_columns.issubset(columns)