import joblib
from sklearn.metrics import accuracy_score, f1_score

from src.ml.generate_data import generate_broadband_data


def main():
    model_data = joblib.load(
        "models/broadband_anomaly_model.joblib"
    )

    model = model_data["model"]
    features = model_data["features"]

    seeds = [101, 202, 303, 404, 505]

    print("\n=== Model Robustness Test ===")
    print(f"{'Seed':<10}{'Accuracy':<15}{'F1 Score':<15}")
    print("-" * 40)

    accuracies = []
    f1_scores = []

    for seed in seeds:
        data = generate_broadband_data(
            rows=2000,
            seed=seed,
        )

        actual = data["is_anomaly"]
        predicted = model.predict(data[features])

        accuracy = accuracy_score(actual, predicted)
        f1 = f1_score(actual, predicted)

        accuracies.append(accuracy)
        f1_scores.append(f1)

        print(f"{seed:<10}{accuracy:<15.4f}{f1:<15.4f}")

    print("-" * 40)
    print(f"Mean accuracy: {sum(accuracies) / len(accuracies):.4f}")
    print(f"Mean F1 score: {sum(f1_scores) / len(f1_scores):.4f}")


if __name__ == "__main__":
    main()