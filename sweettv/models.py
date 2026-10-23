"""Method 1 (content-based TF-IDF) and Method 2 (implicit ALS).

Both expose score(users, items) -> dense matrix, -inf where the method
cannot score (unknown user / unknown item). Ranking, repeat filtering and
fallback are shared and live in recs.py.
"""
import re

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

FIELDS = ("category", "genre", "year", "director", "actors")
WEEK = 7 * 24 * 3600


def _split(s, seps=r"[,/]"):
    if not isinstance(s, str):
        return []
    return [t.strip().lower() for t in re.split(seps, s) if t.strip()]


def _year_tokens(s):
    if not isinstance(s, str):
        return []
    m = re.match(r"\s*(\d{4})", s)
    toks = []
    if m:
        y = int(m.group(1))
        toks.append(f"decade_{y // 10 * 10}")
        toks.append("era_" + ("pre1980" if y < 1980 else "1980_1999" if y < 2000 else "2000_2009" if y < 2010 else "2010_2014" if y < 2015 else "2015plus"))
    if re.search(r"\d{4}\s*-\s*$", s) or "-0000" in s:
        toks.append("running")
    elif re.search(r"\d{4}\s*-\s*\d{4}", s):
        toks.append("ended_series")
    return toks


def item_field_tokens(catalog: pd.DataFrame) -> dict:
    return {
        "category": [[c.lower()] if isinstance(c, str) else [] for c in catalog["tv_show_category"]],
        "genre": [[g.strip().lower() for g in gs if isinstance(g, str)] for gs in
                  zip(catalog["tv_show_genre_1"], catalog["tv_show_genre_2"], catalog["tv_show_genre_3"])],
        "year": [_year_tokens(y) for y in catalog["year_of_production"]],
        "director": [_split(d) for d in catalog["director"]],
        "actors": [_split(a) for a in catalog["actors"]],
    }


def interaction_weights(inter: pd.DataFrame, weighting: str, half_life_weeks, end_sec: int) -> np.ndarray:
    if weighting == "count":
        w = inter["n_airings"].to_numpy(float)
    elif weighting == "log_count":
        w = np.log1p(inter["n_airings"].to_numpy(float))
    elif weighting == "hours":
        w = inter["watch_sec"].to_numpy(float) / 3600
    elif weighting == "log_hours":
        w = np.log1p(inter["watch_sec"].to_numpy(float) / 3600)
    elif weighting == "binary":
        w = np.ones(len(inter))
    else:
        raise ValueError(weighting)
    if half_life_weeks:
        age = (end_sec - inter["last_start"].to_numpy()) / WEEK
        w = w * 0.5 ** (age / half_life_weeks)
    return w


class ContentBased:
    """TF-IDF item vectors per metadata field (weighted, concatenated, L2-normed);
    user profile = weighted sum of watched item vectors; score = cosine."""

    def __init__(self, catalog: pd.DataFrame, field_weights: dict, weighting="log_count", half_life_weeks=None):
        self.field_weights, self.weighting, self.half_life = field_weights, weighting, half_life_weeks
        self.items = catalog["tv_show_id"].to_numpy()
        self.item_pos = pd.Series(np.arange(len(self.items)), index=self.items)
        toks = item_field_tokens(catalog)
        blocks = []
        for f in FIELDS:
            w = field_weights.get(f, 0)
            if w <= 0:
                continue
            vec = TfidfVectorizer(analyzer=lambda x: x, sublinear_tf=True, norm="l2")
            blocks.append(w * vec.fit_transform(toks[f]))
        self.X = normalize(sp.hstack(blocks).tocsr())

    def fit(self, inter: pd.DataFrame, end_sec: int):
        inter = inter[inter["tv_show_id"].isin(self.item_pos.index)]
        self.users = inter["user_id"].unique()
        self.user_pos = pd.Series(np.arange(len(self.users)), index=self.users)
        w = interaction_weights(inter, self.weighting, self.half_life, end_sec)
        R = sp.csr_matrix((w, (self.user_pos[inter["user_id"]].to_numpy(), self.item_pos[inter["tv_show_id"]].to_numpy())),
                          shape=(len(self.users), len(self.items)))
        self.P = normalize(R @ self.X)
        return self

    def score(self, users, items) -> np.ndarray:
        out = np.full((len(users), len(items)), -np.inf, dtype=np.float32)
        ui = self.user_pos.reindex(users).to_numpy()
        ii = self.item_pos.reindex(items).to_numpy()
        ru, ci = np.where(~np.isnan(ui))[0], np.where(~np.isnan(ii))[0]
        if len(ru) and len(ci):
            s = (self.P[ui[ru].astype(int)] @ self.X[ii[ci].astype(int)].T).toarray()
            out[np.ix_(ru, ci)] = s
        return out


class ALS:
    """implicit ALS (CPU) on a user x show confidence matrix."""

    def __init__(self, factors=64, regularization=0.1, alpha=10.0, iterations=20, weighting="log_count",
                 half_life_weeks=None, seed=42, num_threads=8):
        self.params = dict(factors=factors, regularization=regularization, alpha=alpha, iterations=iterations)
        self.weighting, self.half_life, self.seed, self.num_threads = weighting, half_life_weeks, seed, num_threads

    def fit(self, inter: pd.DataFrame, end_sec: int):
        from implicit.cpu.als import AlternatingLeastSquares

        self.users = inter["user_id"].unique()
        self.items = inter["tv_show_id"].unique()
        self.user_pos = pd.Series(np.arange(len(self.users)), index=self.users)
        self.item_pos = pd.Series(np.arange(len(self.items)), index=self.items)
        w = interaction_weights(inter, self.weighting, self.half_life, end_sec).astype(np.float32)
        R = sp.csr_matrix((w, (self.user_pos[inter["user_id"]].to_numpy(), self.item_pos[inter["tv_show_id"]].to_numpy())),
                          shape=(len(self.users), len(self.items)))
        self.model = AlternatingLeastSquares(**self.params, random_state=self.seed, num_threads=self.num_threads,
                                             calculate_training_loss=False)
        self.model.fit(R, show_progress=False)
        return self

    def score(self, users, items) -> np.ndarray:
        out = np.full((len(users), len(items)), -np.inf, dtype=np.float32)
        ui = self.user_pos.reindex(users).to_numpy()
        ii = self.item_pos.reindex(items).to_numpy()
        ru, ci = np.where(~np.isnan(ui))[0], np.where(~np.isnan(ii))[0]
        if len(ru) and len(ci):
            U = self.model.user_factors[ui[ru].astype(int)]
            V = self.model.item_factors[ii[ci].astype(int)]
            out[np.ix_(ru, ci)] = U @ V.T
        return out


def _minmax_rows(s: np.ndarray) -> np.ndarray:
    fin = np.isfinite(s)
    lo = np.where(fin, s, np.inf).min(1, keepdims=True)
    hi = np.where(fin, s, -np.inf).max(1, keepdims=True)
    rng = np.where(hi > lo, hi - lo, 1.0)
    out = np.where(fin, (s - lo) / rng, 0.0)
    out[~fin.any(1)] = -np.inf
    return out


def _rank_rows(s: np.ndarray, k0: float = 60.0) -> np.ndarray:
    """Reciprocal-rank transform per row (unscorable -> 0)."""
    order = np.argsort(-np.where(np.isfinite(s), s, -np.inf), axis=1, kind="mergesort")
    ranks = np.empty_like(order)
    np.put_along_axis(ranks, order, np.arange(s.shape[1])[None, :].repeat(s.shape[0], 0), axis=1)
    out = np.where(np.isfinite(s), 1.0 / (k0 + ranks + 1), 0.0)
    out[~np.isfinite(s).any(1)] = -np.inf
    return out


def fuse(s_a: np.ndarray, s_b: np.ndarray, w_a: float, how: str = "minmax") -> np.ndarray:
    """Hybrid score: w_a * norm(a) + (1 - w_a) * norm(b); a row stays -inf only
    if both methods cannot score that user."""
    f = _minmax_rows if how == "minmax" else _rank_rows
    a, b = f(s_a), f(s_b)
    a_ok, b_ok = np.isfinite(a).all(1, keepdims=True), np.isfinite(b).all(1, keepdims=True)
    out = w_a * np.where(a_ok, a, 0) + (1 - w_a) * np.where(b_ok, b, 0)
    out[~(a_ok | b_ok)[:, 0]] = -np.inf
    return out
