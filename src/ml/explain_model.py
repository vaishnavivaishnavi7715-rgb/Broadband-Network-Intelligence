
from pathlib import Path

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

MODEL_FILES = {
    "Anomaly Detection": (
        MODELS_DIR / "broadband_anomaly_model.joblib"
    ),
    "Issue Classification": (
        MODELS_DIR / "network_issue_classifier.joblib"
    ),
}


def get_feature_importance(model_path, model_name):
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}"
        )

    saved_model = joblib.load(model_path)
    model = saved_model["model"]
    features = saved_model["features"]

    # Support pipelines by accessing the final estimator.
    estimator = model

    if hasattr(model, "named_steps"):
        estimator = list(model.named_steps.values())[-1]

    if not hasattr(estimator, "feature_importances_"):
        raise TypeError(
            f"{model_name} final estimator "
            "does not support feature importance."
        )

    importance = estimator.feature_importances_

    if len(features) != len(importance):
        raise ValueError(
            f"Feature count mismatch in {model_name}."
        )

    result = pd.DataFrame({
        "model": model_name,
        "feature": features,
        "importance": importance,
    })

    return result.sort_values(
        "importance", ascending=False
    )


def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    results = []

    for model_name, model_path in MODEL_FILES.items():
        print(f"\n{'=' * 50}")
        print(model_name)
        print("=" * 50)

        importance_df = get_feature_importance(
            model_path, model_name
        )

        print(
            importance_df[
                ["feature", "importance"]
            ].to_string(
                index=False,
                formatters={
                    "importance": "{:.4f}".format
                },
            )
        )

        results.append(importance_df)

    combined = pd.concat(results, ignore_index=True)

    output_path = REPORTS_DIR / "feature_importance.csv"
    combined.to_csv(output_path, index=False)

    print(f"\nFeature importance report saved to:\n{output_path}")


if __name__ == "__main__":
    main()
