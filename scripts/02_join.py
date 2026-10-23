"""Step 2. Interval join log x schedule -> (user, airing) overlaps, then
"watched" rules -> (user, tv_show_id) interaction tables.

Periods are joined separately with views clipped at the period boundaries,
so nothing after a split point leaks into the period before it.
Outputs: outputs/cache/ua_<period>.parquet, outputs/join/*
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from sweettv import config, data
from sweettv.join import RULES, clip_views, interactions, join_user_airings, watched
from sweettv.plots import SERIES, plt, save

OUT = config.out("join")
OUT.mkdir(parents=True, exist_ok=True)
PERIODS = {
    "valid_train": (config.W11, config.W21),
    "valid_holdout": (config.W21, config.W31),
    "final_train": (config.W11, config.W31),
}

log = data.load_log()
sched = data.load_schedule()
known_channels = set(sched.channel_id)
stats, rule_rows = {}, []
for name, (t0, t1) in PERIODS.items():
    t = time.time()
    views = clip_views(log, t0, t1)
    ua = join_user_airings(views, sched)
    ua.to_parquet(config.CACHE_DIR / f"ua_{name}.parquet", index=False)
    vsec = (views.e - views.s)
    on_known = views.channel_id.isin(known_channels)
    stats[name] = {
        "views": len(views), "view_hours": float(vsec.sum() / 3600),
        "view_hours_on_channels_in_schedule": float(vsec[on_known].sum() / 3600),
        "share_of_mapped_view_time_on_labeled_airings": float(ua.overlap_sec.sum() / vsec[on_known].sum()),
        "user_airing_pairs_labeled": len(ua), "users_with_any_labeled_overlap": int(ua.user_id.nunique()),
        "join_seconds": round(time.time() - t, 1),
    }
    for rule in RULES:
        it = interactions(watched(ua, rule))
        rule_rows.append({"period": name, "rule": rule, "user_airing_watched": int(len(watched(ua, rule))),
                          "user_show_pairs": len(it), "users": it.user_id.nunique(), "shows": it.tv_show_id.nunique(),
                          "median_shows_per_user": float(it.groupby("user_id").size().median())})
    print(name, stats[name])

pd.DataFrame(rule_rows).to_csv(OUT / "interactions_by_rule.csv", index=False)
json.dump(stats, open(OUT / "join_stats.json", "w"), indent=1)

# ---------------- distributions on the validation-train period (weeks 11-20)
ua = pd.read_parquet(config.CACHE_DIR / "ua_valid_train.parquet")
cat = data.items_catalog(sched).set_index("tv_show_id")["tv_show_category"]
ua["category"] = cat.reindex(ua.tv_show_id).values
Q = [.05, .1, .25, .5, .75, .9, .95]
qt = pd.DataFrame({"overlap_sec": ua.overlap_sec.quantile(Q), "frac": ua.frac.quantile(Q)})
qt.to_csv(OUT / "user_airing_overlap_quantiles.csv")
bins = [0, .05, .1, .25, .5, .75, .9, .999, 1.0]
fr = pd.cut(ua.frac, bins, include_lowest=True).value_counts(normalize=True).sort_index()
fr.to_csv(OUT / "frac_histogram.csv")
by_cat = ua.groupby("category").agg(pairs=("frac", "size"), median_frac=("frac", "median"),
                                    share_frac_ge_50=("frac", lambda x: (x >= .5).mean()),
                                    share_overlap_ge_300=("overlap_sec", lambda x: (x >= 300).mean()),
                                    median_airing_min=("duration", lambda x: x.median() / 60))
by_cat.to_csv(OUT / "overlap_by_category.csv")
# agreement between rules
m = pd.DataFrame({r: (ua.overlap_sec >= thr) if k == "abs" else (ua.frac >= thr) for r, (k, thr) in RULES.items()})
agree = pd.crosstab(m["abs300"], m["frac50"], normalize=True)
agree.to_csv(OUT / "abs300_vs_frac50_crosstab.csv")
# short airings vs abs threshold; long airings vs frac threshold
ua["dur_bucket"] = pd.cut(ua.duration / 60, [0, 10, 30, 60, 120, 1000], labels=["<=10m", "10-30m", "30-60m", "1-2h", ">2h"])
bucket = ua.groupby("dur_bucket", observed=True).agg(
    pairs=("frac", "size"), frac50=("frac", lambda x: (x >= .5).mean()),
    abs300=("overlap_sec", lambda x: (x >= 300).mean()), any60=("overlap_sec", lambda x: (x >= 60).mean()))
bucket.to_csv(OUT / "watched_share_by_airing_length.csv")

fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
axes[0].hist(ua.frac, bins=50, color=SERIES[0])
axes[0].axvline(.5, color="#52514e", ls="--", lw=.8)
axes[0].set_title("Share of airing watched, per (user, labeled airing)")
axes[0].set_xlabel("overlap / airing duration"); axes[0].set_ylabel("pairs")
b = np.logspace(0, np.log10(ua.overlap_sec.max()), 60)
axes[1].hist(ua.overlap_sec.clip(lower=1), bins=b, color=SERIES[0])
for x, lbl in [(60, "1 min"), (300, "5 min")]:
    axes[1].axvline(x, color="#52514e", ls="--", lw=.8); axes[1].text(x, axes[1].get_ylim()[1] * .95, f" {lbl}", fontsize=8)
axes[1].set_xscale("log"); axes[1].set_title("Overlap seconds, per (user, labeled airing)")
axes[1].set_xlabel("seconds (log scale)"); axes[1].set_ylabel("pairs")
save(fig, OUT / "overlap_distributions.png")

print(qt.to_string()); print(fr.to_string()); print(by_cat.round(3).to_string())
print(agree.round(4).to_string()); print(bucket.round(3).to_string())
print(pd.DataFrame(rule_rows).to_string())
