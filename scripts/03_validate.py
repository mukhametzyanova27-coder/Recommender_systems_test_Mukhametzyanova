"""Step 3-6 on the validation split (train weeks 11-20, holdout weeks 21-30).

Holdout is used only to choose rules / hyperparameters / fusion weights.
A. "watched" threshold rules: train-rule x holdout-rule matrix
B. repeats of already watched shows: allowed vs filtered, per method
C. popularity baseline recency, D. content-based grid, E. ALS grid,
F. hybrid (score / rank fusion of the two methods)
Outputs: outputs/validation/*.csv, best_config.json, results.csv
"""
import itertools
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from sweettv import config, data
from sweettv.evaluation import Harness
from sweettv.join import RULES
from sweettv.models import ALS, ContentBased, fuse
from sweettv.recs import Split

OUT = config.out("validation")
OUT.mkdir(parents=True, exist_ok=True)
np.random.seed(config.SEED)

sched = data.load_schedule()
tpl = data.load_template()
ua_tr = pd.read_parquet(config.CACHE_DIR / "ua_valid_train.parquet")
ua_ho = pd.read_parquet(config.CACHE_DIR / "ua_valid_holdout.parquet")
catalog = data.items_catalog(sched[sched["start"] < config.W31])  # everything known up to the end of the target window
log = print

# Reference configs used while the rules are being chosen (picked from a coarse pre-scan, re-tuned below).
CB_REF = dict(field_weights={f: 1.0 for f in ("category", "genre", "year", "director", "actors")}, weighting="log_count", half_life_weeks=None)
ALS_REF = dict(factors=64, regularization=10.0, alpha=1.0, iterations=15, weighting="log_count", half_life_weeks=2)


def fit_cb(split, p):
    return ContentBased(catalog, p["field_weights"], p["weighting"], p["half_life_weeks"]).fit(split.inter, split.end_sec)


def fit_als(split, p):
    return ALS(**p, seed=config.SEED).fit(split.inter, split.end_sec)


# ------------------------------------------------------------------ A. rules
splits = {r: Split("valid", r, sched, ua_tr) for r in RULES}
rows = []
for gt_rule in RULES:
    for tr_rule in RULES:
        h = Harness(splits[tr_rule], ua_ho, gt_rule, tpl.user_id)
        cb, als = fit_cb(h.split, CB_REF), fit_als(h.split, ALS_REF)
        r = {"train_rule": tr_rule, "holdout_rule": gt_rule, "eval_users": len(h.users_test),
             "mean_truth_size": np.mean([len(h.truth[u]) for u in h.users_test]),
             "popularity": h.run()["map"], "content_ref": h.run(cb)["map"], "als_ref": h.run(als)["map"]}
        rows.append(r); log(r)
rules_df = pd.DataFrame(rows)
rules_df.to_csv(OUT / "A_threshold_rules.csv", index=False)

# The holdout definition of "watched" is fixed to frac50 (>= 50% of the airing, the valley of the bimodal
# overlap-share distribution, see outputs/join). The training rule is chosen within that column:
# best mean MAP@5 over the three reference methods.
GT_RULE = "frac50"
col = rules_df[rules_df.holdout_rule == GT_RULE].set_index("train_rule")[["popularity", "content_ref", "als_ref"]]
RULE = col.mean(axis=1).idxmax()
log("train rule chosen:", RULE, col.mean(axis=1).to_dict())
split = splits[RULE]
H = Harness(split, ua_ho, GT_RULE, tpl.user_id)
log("eval users (test users with non-empty holdout):", len(H.users_test), "all users:", len(H.users_all))
cold_test = [u for u in H.users_test if u not in split.seen]

# ------------------------------------------------------------------ B. repeats
cb_ref, als_ref = fit_cb(split, CB_REF), fit_als(split, ALS_REF)
seen_share = np.mean([len(H.truth[u] & split.seen.get(u, set())) / len(H.truth[u]) for u in H.users_test])
rows = []
for name, m in [("popularity", None), ("content_ref", cb_ref), ("als_ref", als_ref)]:
    for rep in (True, False):
        r = H.run(m, allow_repeats=rep)
        rows.append({"method": name, "allow_repeats": rep, "map5": r["map"], "se": r["se"]})
rep_df = pd.DataFrame(rows)
rep_df.to_csv(OUT / "B_repeats.csv", index=False)
log(rep_df, "\nshare of holdout shows already watched in train:", seen_share)
ALLOW_REPEATS = bool(rep_df.groupby("allow_repeats").map5.mean().idxmax())
H.allow_repeats = ALLOW_REPEATS

# ------------------------------------------------------------------ C. popularity
rows = []
for rw in (1, 2, 4, None):
    H.set_popularity(rw)
    r = H.run()
    rows.append({"recent_weeks": rw if rw else "all (10)", "map5": r["map"], "se": r["se"]})
pop_df = pd.DataFrame(rows)
pop_df.to_csv(OUT / "C_popularity.csv", index=False)
best_rw = [1, 2, 4, None][int(pop_df.map5.values.argmax())]
H.set_popularity(best_rw)  # the same fallback is then used by every method
log(pop_df)

# ------------------------------------------------------------------ D. content-based
FIELD_SETS = {
    "equal": dict(category=1, genre=1, year=1, director=1, actors=1),
    "cat_genre": dict(category=1, genre=1),
    "cat_genre_year": dict(category=1, genre=1, year=1),
    "people_heavy": dict(category=.5, genre=.5, year=.5, director=1.5, actors=1.5),
    "genre_heavy": dict(category=1, genre=2, year=.5, director=.5, actors=.5),
    "cat_heavy": dict(category=2, genre=1, year=.5, director=.5, actors=.5),
    "people_xheavy": dict(category=.5, genre=.5, year=.5, director=3, actors=3),
    "people_only": dict(category=.25, genre=.25, year=.25, director=2, actors=2),
}
rows = []
for (fs, fw), wt, hl in itertools.product(FIELD_SETS.items(), ("log_count", "count", "log_hours", "binary"), (None, .5, 1, 2, 4)):
    p = dict(field_weights=fw, weighting=wt, half_life_weeks=hl)
    t = time.time(); m = fit_cb(split, p); fit_s = time.time() - t
    r = H.run(m)
    rows.append({"field_set": fs, "weighting": wt, "half_life_weeks": hl, "map5": r["map"], "se": r["se"],
                 "fit_sec": fit_s, "predict_sec": r["predict_sec"]})
cb_df = pd.DataFrame(rows).sort_values("map5", ascending=False)
cb_df.to_csv(OUT / "D_content_grid.csv", index=False)
b = cb_df.iloc[0]
CB_BEST = dict(field_weights=FIELD_SETS[b.field_set], weighting=b.weighting,
               half_life_weeks=None if pd.isna(b.half_life_weeks) else float(b.half_life_weeks))
log(cb_df.head(10))

# ------------------------------------------------------------------ E. ALS
rows = []
grid = itertools.product((32, 64, 128, 256), (1.0, 3.0, 10.0, 30.0, 100.0), (0.1, 0.3, 1.0, 3.0),
                         ("log_count", "count", "log_hours"), (None, 1, 2, 4))
for f, reg, a, wt, hl in grid:
    p = dict(factors=f, regularization=reg, alpha=a, iterations=15, weighting=wt, half_life_weeks=hl)
    t = time.time(); m = fit_als(split, p); fit_s = time.time() - t
    r = H.run(m)
    rows.append({**p, "map5": r["map"], "se": r["se"], "fit_sec": fit_s, "predict_sec": r["predict_sec"]})
als_df = pd.DataFrame(rows).sort_values("map5", ascending=False)
# stage 2: local refinement around the best coarse config
b = als_df.iloc[0]
hl0 = None if pd.isna(b.half_life_weeks) else float(b.half_life_weeks)
local = itertools.product((64, 128, 256), sorted({b.regularization * x for x in (.6, 1, 1.7)}),
                          sorted({round(b.alpha * x, 4) for x in (.1, .3, 1)}),
                          (None, 8, 16) if hl0 is None else sorted({hl0 * x for x in (.75, 1, 1.5, 2)}))
rows = []
for f, reg, a, hl in local:
    p = dict(factors=f, regularization=reg, alpha=a, iterations=15, weighting=b.weighting, half_life_weeks=hl)
    t = time.time(); m = fit_als(split, p); fit_s = time.time() - t
    r = H.run(m)
    rows.append({**p, "map5": r["map"], "se": r["se"], "fit_sec": fit_s, "predict_sec": r["predict_sec"]})
als_local = pd.DataFrame(rows).sort_values("map5", ascending=False)
als_local.to_csv(OUT / "E_als_local.csv", index=False)
cand = pd.concat([als_df.head(5), als_local.head(5)]).drop_duplicates(["factors", "regularization", "alpha", "weighting", "half_life_weeks"])
# stage 3: number of iterations for the best configs of stages 1-2
rows2 = []
for _, b in cand.sort_values("map5", ascending=False).head(5).iterrows():
    for it in (15, 30, 50):
        p = dict(factors=int(b.factors), regularization=b.regularization, alpha=b.alpha, iterations=it,
                 weighting=b.weighting, half_life_weeks=None if pd.isna(b.half_life_weeks) else float(b.half_life_weeks))
        t = time.time(); m = fit_als(split, p); fit_s = time.time() - t
        r = H.run(m)
        rows2.append({**p, "map5": r["map"], "se": r["se"], "fit_sec": fit_s, "predict_sec": r["predict_sec"]})
als_df.to_csv(OUT / "E_als_grid.csv", index=False)
als2_df = pd.DataFrame(rows2).sort_values("map5", ascending=False)
als2_df.to_csv(OUT / "E_als_iterations.csv", index=False)
b = als2_df.iloc[0]
ALS_BEST = dict(factors=int(b.factors), regularization=float(b.regularization), alpha=float(b.alpha), iterations=int(b.iterations),
                weighting=b.weighting, half_life_weeks=None if pd.isna(b.half_life_weeks) else float(b.half_life_weeks))
log(als_df.head(10), als_local.head(10), als2_df.head(5))

# ------------------------------------------------------------------ F. hybrid
cb_best, als_best = fit_cb(split, CB_BEST), fit_als(split, ALS_BEST)
S_cache = {}


def scores(which):
    if which not in S_cache:
        u = H.users(which)
        S_cache[which] = (als_best.score(u, split.pool), cb_best.score(u, split.pool))
    return S_cache[which]


rows = []
for how, w in itertools.product(("minmax", "rrf"), np.round(np.arange(0, 1.01, .1), 2)):
    fn = (lambda users, how=how, w=w: fuse(*scores("test"), w, how))
    r = H.run(fn)
    rows.append({"fusion": how, "w_als": w, "map5": r["map"], "se": r["se"]})
hyb_df = pd.DataFrame(rows).sort_values("map5", ascending=False)
hyb_df.to_csv(OUT / "F_hybrid_grid.csv", index=False)
b = hyb_df.iloc[0]
HYB_BEST = dict(fusion=b.fusion, w_als=float(b.w_als))
log(hyb_df.head(10))

# ------------------------------------------------------------------ results table
t = time.time(); cb_best = fit_cb(split, CB_BEST); cb_fit = time.time() - t
t = time.time(); als_best = fit_als(split, ALS_BEST); als_fit = time.time() - t
S_cache.clear()
methods = {
    "Popularity baseline": (None, {"recent_weeks": best_rw}, 0.0),
    "Method 1: Content-based (TF-IDF)": (cb_best, CB_BEST, cb_fit),
    "Method 2: ALS (implicit)": (als_best, ALS_BEST, als_fit),
    "Hybrid: ALS + CB fusion": ("hybrid", {**HYB_BEST, "als": ALS_BEST, "content": CB_BEST}, cb_fit + als_fit),
}
res_rows, val_recs, aps = [], {}, {}
for name, (m, params, fit_sec) in methods.items():
    row = {"method": name, "params": json.dumps(params, ensure_ascii=False, default=str), "fit_sec": fit_sec}
    for which in ("test", "other", "all"):
        if m == "hybrid":
            m_ = (lambda users, which=which: fuse(*scores(which), HYB_BEST["w_als"], HYB_BEST["fusion"]))
        else:
            m_ = m
        t = time.time(); r = H.run(m_, which=which); sec = time.time() - t
        row[f"map5_{which}"] = r["map"]; row[f"se_{which}"] = r["se"]
        aps[(name, which)] = r["ap"]
        if which == "test":
            row.update(r["stats"])
            row["predict_sec_test_users"] = sec
            val_recs[name] = r["recs"]
    res_rows.append(row)
res = pd.DataFrame(res_rows)


def paired(a, b, which, n_boot=2000):
    """Mean paired AP difference a - b with a bootstrap 95% CI over users."""
    d = aps[(a, which)] - aps[(b, which)]
    idx = np.random.default_rng(config.SEED).integers(0, len(d), size=(n_boot, len(d)))
    lo, hi = np.percentile(d[idx].mean(1), [2.5, 97.5])
    return d.mean(), lo, hi


names = list(methods)
for i, name in enumerate(names):
    for which in ("test", "other"):
        d, lo, hi = paired(name, names[0], which)
        res.loc[i, f"diff_vs_pop_{which}"] = d
        res.loc[i, f"diff_vs_pop_{which}_ci95"] = f"[{lo:.4f}, {hi:.4f}]"
res.to_csv(OUT / "results.csv", index=False)
log(res.to_string())

single = res.iloc[1:3]
best_single = single.loc[single.map5_test.idxmax(), "method"]
hyb_vs_single = {w: paired(names[3], best_single, w) for w in ("test", "other")}
pd.DataFrame([{"comparison": f"{names[3]} - {best_single}", "users": w, "mean_diff": v[0], "ci95_low": v[1], "ci95_high": v[2]}
              for w, v in hyb_vs_single.items()]).to_csv(OUT / "G_hybrid_vs_best_single.csv", index=False)
log("hybrid vs best single:", hyb_vs_single)
# Hybrid only if it beats the best single method both on the tuning users (test) and on the
# confirmation users (other), which were not used for any choice.
use_hybrid = hyb_vs_single["test"][0] > 0 and hyb_vs_single["other"][0] > 0
final = "hybrid" if use_hybrid else ("als" if "ALS" in best_single else "content")
best = {
    "seed": config.SEED, "train_rule": RULE, "holdout_rule": GT_RULE, "allow_repeats": ALLOW_REPEATS, "pool": "labeled shows with >=1 airing starting in the target window",
    "fallback": f"popularity, distinct users in the last {best_rw} training weeks (ties: whole period)",
    "pop_recent_weeks": best_rw, "content": CB_BEST, "als": ALS_BEST, "hybrid": HYB_BEST, "final_strategy": final,
    "eval_users_test": int(len(H.users_test)), "eval_users_other": int(len(H.users_other)), "eval_users_all": int(len(H.users_all)),
    "test_users_cold_in_validation": int(len(cold_test)), "share_holdout_truth_seen_in_train": float(seen_share),
}
json.dump(best, open(OUT / "best_config.json", "w"), indent=1, default=str)
pd.to_pickle({"recs": val_recs, "truth": H.truth, "users_test": H.users_test}, config.CACHE_DIR / "validation_recs.pkl")
log(json.dumps(best, indent=1, default=str))
