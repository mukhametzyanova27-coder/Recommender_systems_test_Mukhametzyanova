"""Rules shared by every method: candidate pool, seen items, popularity
fallback and completion of each list to exactly K unique ids."""
import numpy as np
import pandas as pd

from . import config
from .join import airings_in_window, interactions, watched

WEEK = 7 * 24 * 3600


class Split:
    """Everything a method may use for one split: training interactions
    (past only) and the list of shows on air in the target window."""

    def __init__(self, name: str, rule: str, schedule: pd.DataFrame, ua_train: pd.DataFrame):
        self.name, self.rule = name, rule
        self.t0, self.t1, self.w0, self.w1 = config.SPLITS[name]
        self.ua = watched(ua_train, rule)
        self.inter = interactions(self.ua)
        on_air = airings_in_window(schedule, self.w0, self.w1)
        self.pool = np.sort(on_air["tv_show_id"].unique())
        self.n_airings_in_window = on_air.groupby("tv_show_id").size()
        self.seen = self.inter.groupby("user_id")["tv_show_id"].agg(set).to_dict()
        self.end_sec = int(self.t1.value // 10**9)

    def popularity(self, recent_weeks) -> pd.Series:
        """Distinct users who watched each pool show during the last `recent_weeks`
        of training (None = whole period); ties broken by whole-period users."""
        ua = self.ua
        allp = ua.groupby("tv_show_id")["user_id"].nunique()
        rec = allp if recent_weeks is None else \
            ua[ua["start"] >= self.end_sec - recent_weeks * WEEK].groupby("tv_show_id")["user_id"].nunique()
        df = pd.DataFrame({"recent": rec, "all": allp}).reindex(self.pool).fillna(0)
        df = df.sort_values(["recent", "all"], ascending=False, kind="mergesort")
        return df["recent"] + df["all"] * 1e-6


def complete(ranked: dict, users, fallback: list, seen: dict, allow_repeats: bool, k: int = config.K) -> dict:
    """Top-k per user: model ranking first, then the popularity fallback, unique
    ids, seen items skipped when repeats are not allowed. Users absent from
    `ranked` (cold users / model cannot score) get the pure fallback."""
    out = {}
    for u in users:
        excl = set() if allow_repeats else seen.get(u, set())
        lst, used = [], set()
        for src in (ranked.get(u, []), fallback):
            for i in src:
                if i not in used and i not in excl:
                    lst.append(i); used.add(i)
                    if len(lst) == k:
                        break
            if len(lst) == k:
                break
        out[u] = lst
    return out


def topn_from_scores(scores: np.ndarray, users, items: np.ndarray, seen: dict, allow_repeats: bool, n: int = 50) -> dict:
    """Ranked item lists from a dense score matrix (-inf = cannot score)."""
    s = scores.copy()
    pos = {it: j for j, it in enumerate(items)}
    if not allow_repeats:
        for r, u in enumerate(users):
            idx = [pos[i] for i in seen.get(u, ()) if i in pos]
            s[r, idx] = -np.inf
    n = min(n, s.shape[1])
    part = np.argpartition(-s, n - 1, axis=1)[:, :n]
    out = {}
    for r, u in enumerate(users):
        cols = part[r][np.argsort(-s[r, part[r]], kind="mergesort")]
        cols = cols[np.isfinite(s[r, cols])]
        if len(cols):
            out[u] = list(items[cols])
    return out


def holdout_truth(ua_holdout: pd.DataFrame, rule: str, w0, w1) -> dict:
    """Shows each user watched in the target window (airings starting inside it)."""
    a, b = int(pd.Timestamp(w0).value // 10**9), int(pd.Timestamp(w1).value // 10**9)
    ua = watched(ua_holdout, rule)
    ua = ua[(ua["start"] >= a) & (ua["start"] < b)]
    return ua.groupby("user_id")["tv_show_id"].agg(set).to_dict()
