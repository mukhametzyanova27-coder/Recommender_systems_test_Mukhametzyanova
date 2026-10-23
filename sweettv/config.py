"""Paths, time windows and global constants.

Data location is never hardcoded: set SWEETTV_DATA_DIR to a folder with the
competition CSVs, otherwise the data is fetched with kagglehub.
"""
import os
from pathlib import Path

import pandas as pd

COMPETITION = "sweettv-tv-program-recommender"
SEED = 42
K = 5

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = Path(os.environ.get("SWEETTV_OUTPUT_DIR", PROJECT_ROOT / "outputs"))
CACHE_DIR = OUTPUT_DIR / "cache"

LOG_FILE = "dataset11-30.csv"
SCHEDULE_FILES = {
    "11-20": "export_arh_11-20-final.csv",
    "21-30": "export_arh_21-30-final.csv",
    "31-42": "export_arh_31-42-final.csv",
}
SUBMISSION_TEMPLATE = "submission.csv"

# ISO weeks of 2020 (Monday 00:00 local time, same clock as log and schedule).
W11 = pd.Timestamp("2020-03-09")
W21 = pd.Timestamp("2020-05-18")
W31 = pd.Timestamp("2020-07-27")
W43 = pd.Timestamp("2020-10-19")

# name -> (train_start, train_end, target_start, target_end)
SPLITS = {
    "valid": (W11, W21, W21, W31),  # train weeks 11-20, holdout weeks 21-30
    "final": (W11, W31, W31, W43),  # train weeks 11-30, predict weeks 31-42
}


def data_dir() -> Path:
    env = os.environ.get("SWEETTV_DATA_DIR")
    if env:
        return Path(env)
    import kagglehub

    return Path(kagglehub.competition_download(COMPETITION))


def out(*parts) -> Path:
    p = OUTPUT_DIR.joinpath(*parts)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p
