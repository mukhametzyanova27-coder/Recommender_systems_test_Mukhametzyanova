"""Raw data loading with a parquet cache."""
import numpy as np
import pandas as pd

from . import config


def _cached(name, build):
    path = config.CACHE_DIR / f"{name}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    df = build()
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return df


def load_log() -> pd.DataFrame:
    """Viewing log: one row per viewing interval. user_id is uint64."""

    def build():
        df = pd.read_csv(
            config.data_dir() / config.LOG_FILE,
            dtype={"user_id": "uint64", "vsetv_id": "int32", "duraton": "int64"},
            parse_dates=["start_time", "stop_time"],
        )
        return df.drop_duplicates().reset_index(drop=True)

    return _cached("log", build)


def load_schedule_raw() -> pd.DataFrame:
    """All three schedule files stacked, with parsed start/end and source file."""

    def build():
        parts = []
        for tag, fname in config.SCHEDULE_FILES.items():
            df = pd.read_csv(config.data_dir() / fname, dtype={"year_of_production": str})
            df["file"] = tag
            parts.append(df)
        df = pd.concat(parts, ignore_index=True)
        df["start"] = pd.to_datetime(df["start_time"], format="%d.%m.%Y %H:%M:%S")
        df["end"] = df["start"] + pd.to_timedelta(df["duration"], unit="s")
        return df.drop(columns="start_time")

    return _cached("schedule_raw", build)


def load_schedule() -> pd.DataFrame:
    """Deduplicated airings (channel_id, start) with positive duration, sorted per channel.

    Files are contiguous per channel; the only (channel, start) duplicate is a
    zero-length row at the DST switch, which the duration filter removes.
    """

    def build():
        df = load_schedule_raw()
        df = df[df["duration"] > 0]
        df = df.sort_values(["channel_id", "start", "file"]).drop_duplicates(["channel_id", "start"])
        df = df.reset_index(drop=True)
        df["airing_id"] = np.arange(len(df), dtype=np.int64)
        return df

    return _cached("schedule", build)


def load_template() -> pd.DataFrame:
    return pd.read_csv(config.data_dir() / config.SUBMISSION_TEMPLATE, dtype={"user_id": "uint64", "tv_show_id": str})


def items_catalog(schedule: pd.DataFrame) -> pd.DataFrame:
    """One row per labeled tv_show_id with its (static) metadata and a representative title."""
    lab = schedule[schedule["tv_show_id"] != 0]
    cols = ["tv_show_category", "tv_show_genre_1", "tv_show_genre_2", "tv_show_genre_3",
            "year_of_production", "director", "actors", "tv_show_title"]

    def mode(s):
        s = s.dropna()
        return s.value_counts().index[0] if len(s) else np.nan

    cat = lab.groupby("tv_show_id")[cols].agg(mode)
    cat["first_seen"] = lab.groupby("tv_show_id")["start"].min()
    cat["n_channels"] = lab.groupby("tv_show_id")["channel_id"].nunique()
    return cat.reset_index()
