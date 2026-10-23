"""Interval join of viewing log with the schedule.

A view matches an airing when vsetv_id == channel_id (verified in 01_eda.py)
and the time intervals overlap. Only labeled airings (tv_show_id != 0) are
items; unlabeled airtime is dropped from the item space.
"""
import numpy as np
import pandas as pd

NS = 10**9


def _sec(ts) -> np.ndarray:
    return ts.values.astype("datetime64[s]").astype(np.int64)


def clip_views(log: pd.DataFrame, t0, t1) -> pd.DataFrame:
    """Views restricted to [t0, t1); a view crossing a boundary is cut so no
    later (future) seconds leak into an earlier period."""
    v = log[(log["stop_time"] > t0) & (log["start_time"] < t1)]
    s = np.maximum(_sec(v["start_time"]), int(pd.Timestamp(t0).value // NS))
    e = np.minimum(_sec(v["stop_time"]), int(pd.Timestamp(t1).value // NS))
    out = pd.DataFrame({"user_id": v["user_id"].values, "channel_id": v["vsetv_id"].values, "s": s, "e": e})
    return out[out["e"] > out["s"]].reset_index(drop=True)


def join_user_airings(views: pd.DataFrame, schedule: pd.DataFrame, labeled_only: bool = True) -> pd.DataFrame:
    """Per (user, airing) total overlap in seconds.

    Returns user_id, airing_id, tv_show_id, channel_id, start (sec), duration,
    overlap_sec, frac (= overlap / airing duration, capped at 1).
    """
    sched = schedule.sort_values(["channel_id", "start"])
    ps_all, pe_all = _sec(sched["start"]), _sec(sched["end"])
    parts = []
    by_ch = {ch: idx for ch, idx in sched.groupby("channel_id").indices.items()}
    for ch, g in views.groupby("channel_id"):
        idx = by_ch.get(ch)
        if idx is None:
            continue
        ps, pe = ps_all[idx], pe_all[idx]
        cummax_end = np.maximum.accumulate(pe)
        s, e = g["s"].values, g["e"].values
        lo = np.searchsorted(cummax_end, s, side="right")
        hi = np.searchsorted(ps, e, side="left")
        cnt = np.clip(hi - lo, 0, None)
        if cnt.sum() == 0:
            continue
        vi = np.repeat(np.arange(len(g)), cnt)
        ai = lo[vi] + (np.arange(cnt.sum()) - np.repeat(np.cumsum(cnt) - cnt, cnt))
        ov = np.minimum(e[vi], pe[ai]) - np.maximum(s[vi], ps[ai])
        keep = ov > 0
        parts.append(pd.DataFrame({
            "user_id": g["user_id"].values[vi[keep]],
            "row": idx[ai[keep]],
            "overlap_sec": ov[keep],
        }))
    ua = pd.concat(parts, ignore_index=True)
    meta = sched[["airing_id", "tv_show_id", "channel_id", "duration"]].copy()
    meta["start"] = ps_all
    ua = ua.join(meta.reset_index(drop=True), on="row").drop(columns="row")
    if labeled_only:
        ua = ua[ua["tv_show_id"] != 0]
    ua = (ua.groupby(["user_id", "airing_id"], sort=False)
            .agg(tv_show_id=("tv_show_id", "first"), channel_id=("channel_id", "first"),
                 start=("start", "first"), duration=("duration", "first"), overlap_sec=("overlap_sec", "sum"))
            .reset_index())
    ua["frac"] = np.minimum(ua["overlap_sec"] / ua["duration"], 1.0)
    return ua


# Threshold rules for "watched" on the (user, airing) level.
RULES = {
    "any60": ("abs", 60),        # any overlap of >= 1 minute
    "abs300": ("abs", 300),      # >= 5 minutes of the airing
    "frac50": ("frac", 0.5),     # >= 50% of the airing duration
}


def watched(ua: pd.DataFrame, rule: str) -> pd.DataFrame:
    kind, thr = RULES[rule]
    m = ua["overlap_sec"] >= thr if kind == "abs" else ua["frac"] >= thr
    return ua[m]


def interactions(ua_watched: pd.DataFrame) -> pd.DataFrame:
    """(user, show) table: number of watched airings, watched seconds, last watch time."""
    return (ua_watched.groupby(["user_id", "tv_show_id"])
            .agg(n_airings=("airing_id", "size"), watch_sec=("overlap_sec", "sum"), last_start=("start", "max"))
            .reset_index())


def airings_in_window(schedule: pd.DataFrame, t0, t1) -> pd.DataFrame:
    """Labeled airings that start inside [t0, t1): what is on air in the window."""
    m = (schedule["tv_show_id"] != 0) & (schedule["start"] >= t0) & (schedule["start"] < t1)
    return schedule[m]
