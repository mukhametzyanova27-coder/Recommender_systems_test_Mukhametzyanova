"""Evaluation harness for one split: shared rules + MAP@5 on fixed user sets."""
import time

import numpy as np
import pandas as pd

from . import config
from .metrics import apk
from .recs import Split, complete, holdout_truth, topn_from_scores


class Harness:
    def __init__(self, split: Split, ua_holdout: pd.DataFrame, gt_rule: str, test_users, allow_repeats=True,
                 pop_recent_weeks=2):
        self.split = split
        self.truth = holdout_truth(ua_holdout, gt_rule, split.w0, split.w1)
        self.users_test = np.array([u for u in test_users if u in self.truth], dtype=np.uint64)
        self.users_all = np.array(sorted(self.truth), dtype=np.uint64)
        tset = set(int(u) for u in test_users)
        # holdout users outside the submission template: never used for any choice -> confirmation set
        self.users_other = np.array([u for u in self.users_all if int(u) not in tset], dtype=np.uint64)
        self.allow_repeats = allow_repeats
        self.set_popularity(pop_recent_weeks)

    def set_popularity(self, recent_weeks):
        self.pop_recent_weeks = recent_weeks
        self.fallback = list(self.split.popularity(recent_weeks).index)

    def users(self, which):
        return {"test": self.users_test, "other": self.users_other, "all": self.users_all}[which]

    def ap(self, recs, users):
        return np.array([apk(self.truth[u], recs[u], config.K) for u in users])

    def run(self, model=None, which="test", allow_repeats=None):
        """Returns dict(map, se, recs, stats, predict_sec). model=None -> popularity baseline."""
        rep = self.allow_repeats if allow_repeats is None else allow_repeats
        users = self.users(which)
        t = time.time()
        if model is None:
            ranked = {}
        else:
            S = model.score(users, self.split.pool) if hasattr(model, "score") else model(users)
            ranked = topn_from_scores(S, users, self.split.pool, self.split.seen, rep)
        recs = complete(ranked, users, self.fallback, self.split.seen, rep)
        sec = time.time() - t
        ap = self.ap(recs, users)
        n_model = np.array([min(len(ranked.get(u, [])), config.K) for u in users])
        stats = {"users": len(users), "model_filled_5": int((n_model == config.K).sum()),
                 "model_filled_partly": int(((n_model > 0) & (n_model < config.K)).sum()),
                 "fallback_only": int((n_model == 0).sum())}
        return {"map": float(ap.mean()), "se": float(ap.std(ddof=1) / np.sqrt(len(ap))), "ap": ap,
                "recs": recs, "stats": stats, "predict_sec": sec}
