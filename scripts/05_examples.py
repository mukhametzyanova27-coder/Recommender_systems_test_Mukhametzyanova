"""Step 8. Two example users on the validation split (models trained on
weeks 11-20): Top-5 of every method next to what the user really watched in
weeks 21-30 and the user's history (top shows and categories in weeks 11-20).

Users: an active one (95th percentile of training watch hours among the
evaluated test users) and a low-activity one (5th percentile of training
watch hours among evaluated test users that have some history; fully cold
users get the identical popularity fallback from every method, so they are
only counted, not shown).
Outputs: outputs/examples/examples.md
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from sweettv import config, data
from sweettv.join import watched

OUT = config.out("examples")
OUT.mkdir(parents=True, exist_ok=True)
best = json.load(open(config.OUTPUT_DIR / "validation" / "best_config.json"))
vr = pd.read_pickle(config.CACHE_DIR / "validation_recs.pkl")
recs, truth, users = vr["recs"], vr["truth"], vr["users_test"]

sched = data.load_schedule()
cat = data.items_catalog(sched[sched["start"] < config.W31]).set_index("tv_show_id")
tr = watched(pd.read_parquet(config.CACHE_DIR / "ua_valid_train.parquet"), best["train_rule"])
ho = watched(pd.read_parquet(config.CACHE_DIR / "ua_valid_holdout.parquet"), best["holdout_rule"])
ho = ho[(ho.start >= config.W21.value // 10**9) & (ho.start < config.W31.value // 10**9)]

hours = tr.groupby("user_id").overlap_sec.sum().div(3600).reindex(users).fillna(0)
n_shows = tr.groupby("user_id").tv_show_id.nunique().reindex(users).fillna(0)
active = hours.sort_values().index[int(.95 * (len(hours) - 1))]
warm = hours[n_shows > 0].sort_values(kind="mergesort")
low = warm.index[int(.05 * (len(warm) - 1))]
n_cold = int((n_shows == 0).sum())
why = {
    active: f"active: 95th percentile of training watch hours among evaluated test users ({hours[active]:.0f} h, {int(n_shows[active])} shows in weeks 11-20)",
    low: f"low-activity: 5th percentile of training watch hours among evaluated test users with history ({hours[low]:.1f} h, {int(n_shows[low])} shows in weeks 11-20)",
}


def label(i):
    r = cat.loc[i] if i in cat.index else None
    title = str(r.tv_show_title)[:60] if r is not None else "?"
    return title, (r.tv_show_category if r is not None else "?")


md = ["# Example users (validation: train weeks 11-20, holdout weeks 21-30)\n",
      f"Cold users among the {len(users)} evaluated test users: {n_cold} (no watched labeled programs in weeks 11-20); "
      "every method gives them the same popularity fallback list.\n"]
for u in (active, low):
    md.append(f"\n## user {u}\n\nWhy: {why[u]}\n")
    t = truth[u]
    md.append(f"Holdout truth size: {len(t)} shows.\n")
    for method, rr in recs.items():
        hist = set(tr.loc[tr.user_id == u, "tv_show_id"])
        rows = [{"rank": k + 1, "tv_show_id": int(i), "title": label(i)[0], "category": label(i)[1],
                 "in_history_11_20": "yes" if i in hist else "",
                 "watched_in_holdout": "yes" if i in t else ""} for k, i in enumerate(rr[u])]
        df = pd.DataFrame(rows)
        md.append(f"\n**{method}** (hits in top-5: {df.watched_in_holdout.eq('yes').sum()}/5)\n\n" + df.to_markdown(index=False))
    h = ho[ho.user_id == u].groupby("tv_show_id").agg(hours=("overlap_sec", lambda x: x.sum() / 3600), airings=("airing_id", "size"))
    h = h.sort_values("hours", ascending=False).head(10)
    h["title"], h["category"] = zip(*[label(i) for i in h.index]) if len(h) else ([], [])
    md.append("\n**Actually watched in holdout (top-10 by hours)**\n\n" + h.round(2).to_markdown())
    g = tr[tr.user_id == u].groupby("tv_show_id").agg(hours=("overlap_sec", lambda x: x.sum() / 3600), airings=("airing_id", "size"))
    g = g.sort_values("hours", ascending=False).head(10)
    if len(g):
        g["title"], g["category"] = zip(*[label(i) for i in g.index])
        c = tr[tr.user_id == u].assign(category=lambda d: cat.tv_show_category.reindex(d.tv_show_id).values) \
            .groupby("category").overlap_sec.sum().div(3600).sort_values(ascending=False)
        md.append("\n**History, weeks 11-20: top-10 shows by hours**\n\n" + g.round(2).to_markdown())
        md.append("\n**History, weeks 11-20: hours by category**\n\n" + c.round(1).to_frame("hours").to_markdown())
    else:
        md.append("\n**History, weeks 11-20:** no watched labeled programs (cold user).")
(OUT / "examples.md").write_text("\n".join(md), encoding="utf-8")
print("\n".join(md))
