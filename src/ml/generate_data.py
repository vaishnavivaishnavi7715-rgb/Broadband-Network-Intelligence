
from pathlib import Path

import numpy as np
import pandas as pd


def generate_broadband_data(
    rows: int = 5000,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic broadband telemetry with normal and issue records."""
    rng = np.random.default_rng(seed)
    # Start with normal network conditions.
    latency = rng.normal(35, 8, rows).clip(5, None)
    download = rng.normal(250, 45, rows).clip(10, None)
    upload = rng.normal(40, 10, rows).clip(2, None)
    packet_loss = rng.beta(1, 30, rows) * 3
    signal = rng.normal(-55, 8, rows).clip(-100, -20)
    drops = rng.poisson(0.15, rows)

    # Select 40% of records as synthetic issue cases.
    shuffled = rng.permutation(rows)
    issue_count = int(rows * 0.40)
    issue_indices = shuffled[:issue_count]
    normal_indices = shuffled[issue_count:]

    # Divide issue cases into five distinct groups.
    groups = np.array_split(issue_indices, 5)
    weak_idx, packet_idx, latency_idx, bandwidth_idx, unstable_idx = groups

    # Weak signal
    signal[weak_idx] = rng.uniform(-100, -86, len(weak_idx))
    download[weak_idx] = rng.uniform(5, 45, len(weak_idx))
    upload[weak_idx] = rng.uniform(1, 12, len(weak_idx))

    # Packet loss
    packet_loss[packet_idx] = rng.uniform(3, 15, len(packet_idx))
    drops[packet_idx] += rng.integers(1, 5, len(packet_idx))

    # High latency
    latency[latency_idx] = rng.uniform(100, 300, len(latency_idx))

    # Low bandwidth
    download[bandwidth_idx] = rng.uniform(5, 30, len(bandwidth_idx))
    upload[bandwidth_idx] = rng.uniform(1, 8, len(bandwidth_idx))

    # Unstable connection
    drops[unstable_idx] = rng.integers(4, 15, len(unstable_idx))
    latency[unstable_idx] = rng.uniform(60, 180, len(unstable_idx))

    df = pd.DataFrame({
        "latency_ms": latency.round(2),
        "download_mbps": download.round(2),
        "upload_mbps": upload.round(2),
        "packet_loss_pct": packet_loss.round(2),
        "signal_strength_dbm": signal.round(2),
        "connection_drops": drops,
        "is_anomaly": 0,
    })

    # Only the selected 40% are marked as anomalies.
    df.loc[issue_indices, "is_anomaly"] = 1

    return df


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    output_path = project_root / "data" / "broadband_telemetry.csv"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = generate_broadband_data()
    df.to_csv(output_path, index=False)

    print(f"Dataset saved to: {output_path}")
    print(f"Total records: {len(df)}")
    print(f"Normal records: {(df['is_anomaly'] == 0).sum()}")
    print(f"Anomalies: {(df['is_anomaly'] == 1).sum()}")
    print("\nSample records:")
    print(df.head())
