from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parent.parent
REPORT_DIR = ROOT / "reports"
matrix = [
    [1251, 49],
    [24, 676],
]
df = pd.DataFrame(
    matrix,
    index=["Normal", "Anomaly"],
    columns=["Predicted Normal", "Predicted Anomaly"],
)
df.index.name = "Actual"
df.to_csv(REPORT_DIR / "candidate_holdout_confusion_matrix.csv")
print("Confusion matrix CSV fixed.")
