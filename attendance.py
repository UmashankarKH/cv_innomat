from datetime import datetime
from pathlib import Path

import pandas as pd

CSV_PATH = Path("data/attendance.csv")
COLUMNS = ["Name", "Date", "Time"]


def load():
    if CSV_PATH.exists():
        return pd.read_csv(CSV_PATH)
    return pd.DataFrame(columns=COLUMNS)


def mark(name):
    """Mark attendance once per person per day. Returns (newly_marked, time_string)."""
    df = load()
    now = datetime.now()
    today = now.strftime("%Y-%m-%d")
    existing = df[(df["Name"] == name) & (df["Date"] == today)]
    if not existing.empty:
        return False, existing.iloc[0]["Time"]
    t = now.strftime("%H:%M:%S")
    df.loc[len(df)] = [name, today, t]
    df.to_csv(CSV_PATH, index=False)
    return True, t
