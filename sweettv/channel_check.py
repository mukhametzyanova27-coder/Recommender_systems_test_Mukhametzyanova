"""Statistical check that vsetv_id (log) and channel_id (schedule) are the same channel ids.

Idea: viewers switch channels at program boundaries, so view start/stop events
of channel X should concentrate right after the boundaries of X's schedule.
Events are binned on a 5-minute grid ([t-30s, t+90s) around each grid point).
Expected counts come from a multiplicative model (date-hour activity x
5-minute-slot-of-hour effect), which removes both day-to-day activity changes
and the generic ":00/:30 switching" effect. For a (log channel, schedule
channel) pair, z = (O - E) / sqrt(E) over the grid points where the schedule
channel has a program start. The true counterpart should have a large z, and
for the true counterpart the z profile over time shifts should peak at 0.
"""
import numpy as np
import pandas as pd

from .join import _sec

STEP = 300


def _grid(t0, t1):
    T0, T1 = int(pd.Timestamp(t0).value // 10**9), int(pd.Timestamp(t1).value // 10**9)
    return T0, T1, (T1 - T0) // STEP


def event_counts(log: pd.DataFrame, t0, t1) -> dict:
    T0, _, G = _grid(t0, t1)
    out = {}
    for ch, g in log.groupby("vsetv_id"):
        x = np.concatenate([_sec(g["start_time"]), _sec(g["stop_time"])]) - T0 + 30
        x = x[(x >= 0) & (x % STEP < 120)]
        t = x // STEP
        out[ch] = np.bincount(t[t < G], minlength=G).astype(float)
    return out


def boundary_flags(schedule: pd.DataFrame, t0, t1, shift: int = 0) -> dict:
    T0, T1, G = _grid(t0, t1)
    out = {}
    for ch, g in schedule.groupby("channel_id"):
        b = _sec(g["start"]) + shift
        b = b[(b >= T0) & (b < T1) & ((b - T0) % STEP == 0)]
        f = np.zeros(G, bool)
        f[(b - T0) // STEP] = True
        out[ch] = f
    return out


def expected(cnt: np.ndarray) -> np.ndarray:
    m = cnt.reshape(-1, 3600 // STEP)
    return (np.outer(m.sum(1), m.sum(0)) / max(m.sum(), 1)).ravel()


def zscore(cnt, exp, flag) -> float:
    e = exp[flag].sum()
    return float((cnt[flag].sum() - e) / np.sqrt(e)) if e > 0 else np.nan


def cross_match(counts: dict, flags_by_shift: dict, titles: pd.Series, min_events: int = 500) -> pd.DataFrame:
    """For every log channel: rank of the same-id schedule channel among all
    schedule channels by max z over a few small schedule shifts (EPG times of
    some channels are a few minutes off)."""
    rows = []
    for v, cnt in counts.items():
        if cnt.sum() < min_events:
            continue
        exp = expected(cnt)
        z = pd.DataFrame({s: {c: zscore(cnt, exp, f) for c, f in fl.items()} for s, fl in flags_by_shift.items()})
        zs = z.max(axis=1).dropna().sort_values(ascending=False)
        rows.append({
            "vsetv_id": v, "in_schedule": v in z.index, "title_same_id": titles.get(v),
            "events": int(cnt.sum()), "z_identity_shift0": z.loc[v, 0] if v in z.index else np.nan,
            "z_identity": zs.get(v, np.nan),
            "identity_best_shift_min": (z.loc[v].idxmax() // 60) if v in z.index else np.nan,
            "identity_rank": (zs.index.get_loc(v) + 1) if v in zs.index else np.nan,
            "n_candidates": len(zs), "best_match": zs.index[0], "best_title": titles.get(zs.index[0]),
            "z_best": zs.iloc[0], "z_second": zs.iloc[1],
        })
    return pd.DataFrame(rows).sort_values("events", ascending=False)


def shift_profile(counts: dict, schedule: pd.DataFrame, t0, t1, channels, shifts) -> pd.DataFrame:
    rows = {}
    flags_by_shift = {s: boundary_flags(schedule[schedule["channel_id"].isin(channels)], t0, t1, s) for s in shifts}
    for v in channels:
        if v not in counts:
            continue
        exp = expected(counts[v])
        rows[v] = {s // 60: zscore(counts[v], exp, flags_by_shift[s][v]) for s in shifts if v in flags_by_shift[s]}
    return pd.DataFrame(rows).T
