
from pathlib import Path

import joblib
import pandas as pd

# Project paths
BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "models" / "broadband_anomaly_model.joblib"
DATA_PATH = BASE_DIR / "data" / "broadband_telemetry.csv"


def load_saved_model():
    """Load the saved model and its feature list."""
    assert MODEL_PATH.exists(), f"Model file not found: {MODEL_PATH}"

    saved_data = joblib.load(MODEL_PATH)

    assert isinstance(saved_data, dict), "Saved model must be a dictionary"
    assert "model" in saved_data, "Model key is missing"
    assert "features" in saved_data, "Features key is missing"

    return saved_data["model"], saved_data["features"]


def test_model_file_exists():
    assert MODEL_PATH.exists(), "Model file does not exist"


def test_dataset_exists():
    assert DATA_PATH.exists(), "Telemetry dataset does not exist"


def test_model_loads():
    model, features = load_saved_model()

    assert model is not None
    assert len(features) > 0


def test_predictions_are_valid():
    model, features = load_saved_model()

    assert DATA_PATH.exists(), "Telemetry dataset does not exist"
    df = pd.read_csv(DATA_PATH)

    assert len(df) > 0, "Dataset is empty"

    missing_features = [name for name in features if name not in df.columns]
    assert not missing_features, f"Missing features: {missing_features}"

    X = df[features].head(10)
    predictions = model.predict(X)

    assert len(predictions) == len(X)
    assert set(predictions).issubset({0, 1}), (
        f"Unexpected prediction values: {set(predictions)}"
    )


def test_prediction_probabilities():
    model, features = load_saved_model()
    df = pd.read_csv(DATA_PATH)
    X = df[features].head(10)

    probabilities = model.predict_proba(X)

    assert probabilities.shape == (len(X), 2)
    assert (probabilities >= 0).all()
    assert (probabilities <= 1).all()
    assert (abs(probabilities.sum(axis=1) - 1) < 1e-6).all()
