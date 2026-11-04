
import requests

API_URL = "http://127.0.0.1:8000"


def test_api_root():
    response = requests.get(f"{API_URL}/", timeout=10)
    assert response.status_code == 200


def test_api_health():
    response = requests.get(f"{API_URL}/health", timeout=10)

    assert response.status_code == 200

    data = response.json()
    assert data.get("status") == "healthy"
    assert data.get("model_loaded") is True


def test_prediction_endpoint():
    payload = {
        "latency_ms": 250,
        "download_mbps": 5,
        "upload_mbps": 1,
        "packet_loss_pct": 5,
        "signal_strength_dbm": -100,
        "connection_drops": 8,
    }

    response = requests.post(
        f"{API_URL}/predict",
        json=payload,
        timeout=10,
    )

    assert response.status_code == 200

    result = response.json()
    assert isinstance(result, dict)
    assert len(result) > 0


def test_prediction_endpoint_rejects_invalid_input():
    response = requests.post(
        f"{API_URL}/predict",
        json={},
        timeout=10,
    )

    assert response.status_code == 422
