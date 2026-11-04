import sqlite3
from pathlib import Path

db = Path("data/prediction_history.db")
print("Database exists:", db.exists())

if db.exists():
    with sqlite3.connect(db) as conn:
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()

        print("Tables:", tables)

        if ("prediction_history",) in tables:
            count = conn.execute(
                "SELECT COUNT(*) FROM prediction_history"
            ).fetchone()[0]
            print("Saved predictions:", count)
        else:
            print("Prediction history table not found.")