import sqlite3
from pathlib import Path

import requests


API_URL = "http://127.0.0.1:8000"
DB_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "prediction_history.db"
)


def test_api_prediction_is_saved_to_history():
    payload = {
        "latency_ms": 250,
        "download_mbps": 5,
        "upload_mbps": 1,
        "packet_loss_pct": 5,
        "signal_strength_dbm": -100,
        "connection_drops": 8,
    }

    with sqlite3.connect(DB_PATH) as conn:
        before = conn.execute(
            "SELECT COUNT(*) FROM prediction_history"
        ).fetchone()[0]

    response = requests.post(
        f"{API_URL}/predict",
        json=payload,
        timeout=10,
    )
    assert response.status_code == 200

    with sqlite3.connect(DB_PATH) as conn:
        after = conn.execute(
            "SELECT COUNT(*) FROM prediction_history"
        ).fetchone()[0]

    assert after == before + 1
