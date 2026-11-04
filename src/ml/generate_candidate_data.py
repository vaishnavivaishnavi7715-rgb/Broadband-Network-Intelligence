
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = PROJECT_ROOT / "data" / "broadband_candidate_telemetry.csv"


def generate_candidate_data(rows: int = 10000, seed: int = 2026):
    rng = np.random.default_rng(seed)

    # Generate a broad range of baseline network measurements.
    latency = rng.normal(35, 15, rows).clip(5, 300)
    download = rng.normal(220, 65, rows).clip(5, 1000)
    upload = rng.normal(35, 12, rows).clip(1, 200)
    packet_loss = rng.beta(1.2, 25, rows) * 5
    signal = rng.normal(-58, 10, rows).clip(-100, -20)
    drops = rng.poisson(0.25, rows)

    # Begin with every row labelled normal.
    labels = np.zeros(rows, dtype=int)

    # Allocate a range of issue severities.
    indices = rng.permutation(rows)
    issue_count = int(rows * 0.35)
    issue_indices = indices[:issue_count]

    groups = np.array_split(issue_indices, 5)
    weak_idx, packet_idx, latency_idx, bandwidth_idx, unstable_idx = groups

    # Weak signal: include a mixture of isolated and severe cases.
    weak_count = len(weak_idx)
    weak_signal_only = weak_idx[: weak_count // 3]
    weak_severe = weak_idx[weak_count // 3 :]

    signal[weak_signal_only] = rng.uniform(-84, -76, len(weak_signal_only))
    signal[weak_severe] = rng.uniform(-100, -86, len(weak_severe))
    download[weak_severe] = rng.uniform(5, 45, len(weak_severe))
    upload[weak_severe] = rng.uniform(1, 12, len(weak_severe))

    # Packet loss: include moderate and severe loss.
    packet_count = len(packet_idx)
    packet_moderate = packet_idx[: packet_count // 3]
    packet_severe = packet_idx[packet_count // 3 :]

    packet_loss[packet_moderate] = rng.uniform(1.5, 3.0, len(packet_moderate))
    packet_loss[packet_severe] = rng.uniform(4, 15, len(packet_severe))
    drops[packet_severe] += rng.integers(1, 5, len(packet_severe))

    # High latency: distinguish moderate from severe latency.
    latency_count = len(latency_idx)
    latency_moderate = latency_idx[: latency_count // 3]
    latency_severe = latency_idx[latency_count // 3 :]

    latency[latency_moderate] = rng.uniform(60, 95, len(latency_moderate))
    latency[latency_severe] = rng.uniform(120, 300, len(latency_severe))

    # Low bandwidth: preserve some overlap with normal measurements.
    download[bandwidth_idx] = rng.uniform(5, 35, len(bandwidth_idx))
    upload[bandwidth_idx] = rng.uniform(1, 8, len(bandwidth_idx))

    # Unstable connection: combine repeated drops and elevated latency.
    drops[unstable_idx] = rng.integers(4, 15, len(unstable_idx))
    latency[unstable_idx] = rng.uniform(60, 180, len(unstable_idx))

    # Label clear issue cases. Some moderate measurements intentionally
    # overlap with normal conditions to expose classification uncertainty.
    labels[issue_indices] = 1

    df = pd.DataFrame(
        {
            "latency_ms": latency.round(2),
            "download_mbps": download.round(2),
            "upload_mbps": upload.round(2),
            "packet_loss_pct": packet_loss.round(2),
            "signal_strength_dbm": signal.round(2),
            "connection_drops": drops,
            "is_anomaly": labels,
        }
    )

    return df


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    df = generate_candidate_data()
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"Candidate dataset saved to: {OUTPUT_PATH}")
    print(f"Total records: {len(df)}")
    print(f"Normal records: {(df['is_anomaly'] == 0).sum()}")
    print(f"Anomaly records: {(df['is_anomaly'] == 1).sum()}")
    print("\nSample records:")
    print(df.head())


if __name__ == "__main__":
    main()
