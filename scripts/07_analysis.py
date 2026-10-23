"""Step 10. Analysis for the business / algorithmic discussion (validation
split, models trained on weeks 11-20, evaluated on weeks 21-30; plus the
final submissions for weeks 31-42).

* familiar vs new: share of recommended slots the user already watched in
  training, and how often familiar / new slots are hits in the holdout
* MAP@5 by user activity in training (number of watched shows)
* catalog coverage and concentration on the most popular shows
* hit rate of recommended slots by program category; users with AP@5 = 0
* how much of the holdout truth a collaborative model can score at all

Outputs: outputs/analysis/*.csv and outputs/analysis/analysis.md
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from sweettv import config, data
from sweettv.join import watched
from sweettv.metrics import apk
from sweettv.recs import Split

OUT = config.out("analysis")
OUT.mkdir(parents=True, exist_ok=True)
best = json.load(open(config.OUTPUT_DIR / "validation" / "best_config.json"))
vr = pd.read_pickle(config.CACHE_DIR / "validation_recs.pkl")
recs, truth, users = vr["recs"], vr["truth"], [u for u in vr["users_test"]]
sched = data.load_schedule()
cat = data.items_catalog(sched).set_index("tv_show_id")["tv_show_category"]
split = Split("valid", best["train_rule"], sched, pd.read_parquet(config.CACHE_DIR / "ua_valid_train.parquet"))
seen = split.seen
pop_rank = {i: r for r, i in enumerate(split.popularity(best["pop_recent_weeks"]).index)}
md = ["# Analysis (validation: train weeks 11-20, holdout weeks 21-30, evaluated test users)\n"]


def table(title, df):
    df.to_csv(OUT / f"{title}.csv", index=False)
    md.append(f"\n## {title}\n\n" + df.round(4).to_markdown(index=False))


# ---------------------------------------------------------------- familiar vs new
rows = []
for m, rr in recs.items():
    fam = hit_fam = new = hit_new = 0
    for u in users:
        s, t = seen.get(u, set()), truth[u]
        for i in rr[u]:
            if i in s:
                fam += 1; hit_fam += i in t
            else:
                new += 1; hit_new += i in t
    n = fam + new
    rows.append({"method": m, "slots": n, "share_familiar": fam / n, "hit_rate_familiar": hit_fam / max(fam, 1),
                 "share_new": new / n, "hit_rate_new": hit_new / max(new, 1), "new_hits_total": hit_new,
                 "familiar_hits_total": hit_fam})
table("familiar_vs_new_validation", pd.DataFrame(rows))
tr_share = [len(truth[u] & seen.get(u, set())) / len(truth[u]) for u in users]
truth_items = [i for u in users for i in truth[u]]
known_items = set(split.inter.tv_show_id)
table("holdout_truth_composition", pd.DataFrame([{
    "users": len(users), "mean_truth_size": np.mean([len(truth[u]) for u in users]),
    "mean_share_of_truth_watched_in_train_by_same_user": np.mean(tr_share),
    "share_of_truth_pairs_on_shows_with_no_train_interactions": np.mean([i not in known_items for i in truth_items]),
    "pool_size": len(split.pool), "pool_shows_with_train_interactions": int(np.isin(split.pool, list(known_items)).sum()),
}]))

# ---------------------------------------------------------------- activity buckets
n_hist = pd.Series({u: len(seen.get(u, ())) for u in users})
bins = [-1, 0, 10, 30, 60, 10**6]
labels = ["0 (cold)", "1-10", "11-30", "31-60", ">60"]
bucket = pd.cut(n_hist, bins, labels=labels)
rows = []
for b in labels:
    us = list(bucket[bucket == b].index)
    r = {"train_shows_bucket": b, "users": len(us)}
    for m, rr in recs.items():
        r[m] = np.mean([apk(truth[u], rr[u]) for u in us]) if us else np.nan
    rows.append(r)
table("map5_by_activity_validation", pd.DataFrame(rows))

# ---------------------------------------------------------------- coverage / concentration
rows = []
for m, rr in recs.items():
    all_ids = [i for u in users for i in rr[u]]
    rows.append({"method": m, "distinct_ids": len(set(all_ids)),
                 "share_slots_in_top10_popular": np.mean([pop_rank.get(i, 10**9) < 10 for i in all_ids]),
                 "share_users_with_identical_list_to_popularity": np.mean([rr[u] == recs["Popularity baseline"][u] for u in users]),
                 "share_users_ap_zero": np.mean([apk(truth[u], rr[u]) == 0 for u in users])})
table("coverage_validation", pd.DataFrame(rows))

# ---------------------------------------------------------------- categories
rows = []
for m, rr in recs.items():
    c = pd.DataFrame([{"category": cat.get(i, "?"), "hit": i in truth[u]} for u in users for i in rr[u]])
    g = c.groupby("category").hit.agg(["size", "mean"]).rename(columns={"size": "slots", "mean": "hit_rate"})
    g["share_of_slots"] = g.slots / g.slots.sum()
    for k, r in g.iterrows():
        rows.append({"method": m, "category": k, "slots": int(r.slots), "share_of_slots": r.share_of_slots, "hit_rate": r.hit_rate})
table("recs_by_category_validation", pd.DataFrame(rows))
tc = pd.Series([cat.get(i, "?") for i in truth_items]).value_counts(normalize=True).rename_axis("category").reset_index(name="share_of_truth_pairs")
table("holdout_truth_by_category", tc)

# users with AP=0 under the final strategy
fin = {"hybrid": "Hybrid: ALS + CB fusion", "als": "Method 2: ALS (implicit)", "content": "Method 1: Content-based (TF-IDF)"}[best["final_strategy"]]
ap = pd.Series({u: apk(truth[u], recs[fin][u]) for u in users})
def films_share(u):
    return np.mean([cat.get(i) == "Фильмы" for i in truth[u]])
grp = pd.DataFrame({"ap_zero": ap == 0, "truth_size": [len(truth[u]) for u in ap.index],
                    "train_shows": n_hist.reindex(ap.index).values, "truth_share_films": [films_share(u) for u in ap.index]})
table("final_strategy_ap_zero_profile", grp.groupby("ap_zero").agg(users=("truth_size", "size"), median_truth_size=("truth_size", "median"),
      median_train_shows=("train_shows", "median"), mean_truth_share_films=("truth_share_films", "mean")).reset_index())

# ---------------------------------------------------------------- what the popularity list is made of
pop = split.popularity(best["pop_recent_weeks"]).head(5)
title = data.items_catalog(sched[sched["start"] < config.W31]).set_index("tv_show_id")["tv_show_title"]
table("popularity_top5_validation", pd.DataFrame({
    "tv_show_id": pop.index, "title": title.reindex(pop.index).values, "category": cat.reindex(pop.index).values,
    "users_last_weeks": pop.values.round().astype(int),
    "share_of_test_users_with_it_in_truth": [np.mean([i in truth[u] for u in users]) for i in pop.index]}))

# ---------------------------------------------------------------- airing frequency and metadata distinctness
na = split.n_airings_in_window
freq = pd.DataFrame({"category": cat.reindex(na.index).values, "airings": na.values})
table("airings_in_holdout_window_by_category", freq.groupby("category").airings.agg(
    shows="size", median_airings="median", share_single_airing=lambda x: (x == 1).mean()).reset_index())
full = data.items_catalog(sched[sched["start"] < config.W31]).set_index("tv_show_id").reindex(split.pool)
sig_cols = ["tv_show_category", "tv_show_genre_1", "tv_show_genre_2", "tv_show_genre_3", "year_of_production", "director", "actors"]
sig = full[sig_cols].astype(str).agg("|".join, axis=1)
dup = sig.map(sig.value_counts()) > 1
table("pool_metadata_distinctness_by_category", pd.DataFrame({"category": full.tv_show_category, "shared": dup.values})
      .groupby("category").shared.agg(shows="size", share_with_identical_metadata="mean").reset_index())

# ---------------------------------------------------------------- final submissions (weeks 31-42)
fsplit = Split("final", best["train_rule"], sched, pd.read_parquet(config.CACHE_DIR / "ua_final_train.parquet"))
rows = []
for m in ("popularity", "content", "als", "hybrid"):
    sub = pd.read_csv(config.OUTPUT_DIR / "final" / f"submission_{m}.csv", dtype={"user_id": "uint64", "tv_show_id": str})
    ids = [(u, int(i)) for u, s in zip(sub.user_id, sub.tv_show_id) for i in s.split()]
    rows.append({"method": m, "distinct_ids": len({i for _, i in ids}),
                 "share_slots_already_watched_weeks_11_30": np.mean([i in fsplit.seen.get(u, set()) for u, i in ids]),
                 "share_slots_by_category": json.dumps(pd.Series([cat.get(i, "?") for _, i in ids]).value_counts(normalize=True).round(3).to_dict(), ensure_ascii=False)})
table("final_submissions_profile", pd.DataFrame(rows))

(OUT / "analysis.md").write_text("\n".join(md), encoding="utf-8")
print("\n".join(md))
