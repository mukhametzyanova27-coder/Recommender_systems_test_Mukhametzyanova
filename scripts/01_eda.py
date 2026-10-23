"""Step 1. EDA: sizes, date ranges, vsetv_id <-> channel_id check, duration
distributions, test-user coverage, submission template format.

Outputs: outputs/eda/*.csv|json|png and outputs/eda/eda_report.md
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from sweettv import config, data
from sweettv import channel_check as cc
from sweettv.plots import SERIES, plt, save

OUT = config.out("eda")
OUT.mkdir(parents=True, exist_ok=True)
Q = [0, .01, .05, .1, .25, .5, .75, .9, .95, .99, .999, 1]
rep = []  # markdown report lines
summary = {}


def section(title):
    rep.append(f"\n## {title}\n")


def table(df):
    rep.append(df.to_markdown() if hasattr(df, "to_markdown") else str(df))


log = data.load_log()
raw = data.load_schedule_raw()
sched = data.load_schedule()
tpl = data.load_template()

# ---------------------------------------------------------------- sizes, dates
section("Sizes and date ranges")
raw_log_rows = sum(1 for _ in open(config.data_dir() / config.LOG_FILE, encoding="utf-8")) - 1
summary["log"] = {
    "rows_raw": raw_log_rows, "rows_after_dedup": len(log), "exact_duplicates_removed": raw_log_rows - len(log),
    "users": int(log.user_id.nunique()), "vsetv_ids": int(log.vsetv_id.nunique()),
    "start_min": str(log.start_time.min()), "start_max": str(log.start_time.max()),
    "stop_max": str(log.stop_time.max()),
}
d = (log.stop_time - log.start_time).dt.total_seconds()
summary["log"]["share_duraton_equals_stop_minus_start"] = float((d == log.duraton).mean())
summary["log"]["rows_where_stop_minus_start_minus_duraton_is_3600"] = int(((d - log.duraton) == 3600).sum())
summary["log"]["min_duraton"] = int(log.duraton.min())
rows = []
for tag, g in raw.groupby("file"):
    rows.append({"file": config.SCHEDULE_FILES[tag], "rows": len(g), "channels": g.channel_id.nunique(),
                 "start_min": g.start.min(), "start_max": g.start.max(),
                 "labeled_rows_share": round((g.tv_show_id != 0).mean(), 4),
                 "labeled_airtime_share": round(g.loc[g.tv_show_id != 0, "duration"].sum() / g.duration.sum(), 4),
                 "unique_tv_show_ids": g.loc[g.tv_show_id != 0, "tv_show_id"].nunique()})
sched_files = pd.DataFrame(rows).set_index("file")
sched_files.to_csv(OUT / "schedule_files.csv")
rep.append("Log:\n\n```\n" + json.dumps(summary["log"], indent=1, ensure_ascii=False) + "\n```\n")
table(sched_files)
rep.append("\nThe schedule files are contiguous per channel (each file ends where the next begins, per broadcast day);"
           f" after stacking, (channel_id, start) duplicates: {int(raw.duplicated(['channel_id', 'start']).sum())}"
           f" (zero-duration DST rows dropped: {int((raw.duration == 0).sum())}).")
s = sched.sort_values(["channel_id", "start"])
gap = (s.start.shift(-1) - s.end).dt.total_seconds()[s.channel_id.shift(-1) == s.channel_id]
summary["schedule_gap_share_zero"] = float((gap == 0).mean())
rep.append(f"\nShare of consecutive airings on a channel with no gap/overlap: {summary['schedule_gap_share_zero']:.5f}")

weekly = log.assign(week=log.start_time.dt.isocalendar().week).groupby("week").agg(
    views=("user_id", "size"), users=("user_id", "nunique"), hours=("duraton", lambda x: round(x.sum() / 3600)))
weekly.to_csv(OUT / "log_weekly.csv")
rep.append("\nViews per ISO week:\n")
table(weekly.T)

# -------------------------------------------------------------- channel mapping
section("vsetv_id (log) vs channel_id (schedule)")
vs, ch = set(log.vsetv_id.unique()), set(sched.channel_id.unique())
missing = sorted(vs - ch)
share_missing_views = float(log.vsetv_id.isin(missing).mean())
share_missing_time = float(log.loc[log.vsetv_id.isin(missing), "duraton"].sum() / log.duraton.sum())
summary["mapping"] = {"vsetv_ids": len(vs), "channel_ids": len(ch), "ids_in_both": len(vs & ch),
                      "vsetv_ids_absent_in_schedule": [int(x) for x in missing],
                      "views_share_on_absent": share_missing_views, "time_share_on_absent": share_missing_time}
titles = sched.groupby("channel_id").channel_title.agg(lambda x: x.value_counts().index[0])
hist = sched[sched.start < config.W31]
counts = cc.event_counts(log, config.W11, config.W31)
flags = {sh: cc.boundary_flags(hist, config.W11, config.W31, sh) for sh in (-600, -300, 0, 300, 600)}
cm = cc.cross_match(counts, flags, titles)
cm.to_csv(OUT / "channel_mapping_check.csv", index=False)
top = cm[cm.in_schedule].head(15)
t = cm[cm.in_schedule]
summary["mapping"].update({
    "tested_channels": int(len(t)),
    "identity_is_best_of_all_schedule_channels": int((t.identity_rank == 1).sum()),
    "events_share_of_tested_where_identity_best": float(t.loc[t.identity_rank == 1, "events"].sum() / t.events.sum()),
    "median_z_best_when_identity_not_best": float(t.loc[t.identity_rank != 1, "z_best"].median()),
    "max_z_best_when_identity_not_best": float(t.loc[t.identity_rank != 1, "z_best"].max()),
    "identity_best_shift_min_counts_where_identity_best": t.loc[t.identity_rank == 1, "identity_best_shift_min"].value_counts().to_dict(),
})
shifts = list(range(-3 * 3600, 3 * 3600 + 1, 900)) + [-600, -300, 300, 600]
prof = cc.shift_profile(counts, hist, config.W11, config.W31, list(cm[cm.in_schedule].vsetv_id.head(20)), sorted(set(shifts)))
prof.insert(0, "title", titles.reindex(prof.index).values)
prof.to_csv(OUT / "channel_shift_profile_top20.csv")
rep.append(f"vsetv_ids in log: {len(vs)}, channel_ids in schedule: {len(ch)}, same ids: {len(vs & ch)}. "
           f"Absent in schedule: {missing} -> {share_missing_views:.2%} of views, {share_missing_time:.2%} of viewing time; "
           "they cannot be joined and are dropped.\n")
rep.append("Boundary-alignment test (see sweettv/channel_check.py). For each log channel, z-score of view start/stop "
           "events right after program starts of every schedule channel (404 candidates, max over schedule shifts "
           "-10..+10 min). Top-15 channels by events:\n")
table(top[["vsetv_id", "title_same_id", "events", "z_identity_shift0", "z_identity", "identity_best_shift_min", "identity_rank",
           "best_match", "best_title", "z_best", "z_second"]].round(1).set_index("vsetv_id"))
rep.append("\nz-score of the identity pair vs time shift of the schedule (minutes), top-20 channels:\n")
table(prof.round(1))
rep.append("\n```\n" + json.dumps(summary["mapping"], indent=1) + "\n```\n")

# ------------------------------------------------------------- durations
section("Duration distributions")
vq = log.duraton.quantile(Q)
lab = sched[sched.tv_show_id != 0]
dq = pd.DataFrame({"view_duration_sec": vq, "airing_duration_all_sec": sched.duration.quantile(Q),
                   "airing_duration_labeled_sec": lab.duration.quantile(Q)})
dq.to_csv(OUT / "duration_quantiles.csv")
table(dq)
long_ = {h: float((log.duraton > h * 3600).mean()) for h in (3, 6, 12, 24)}
summary["view_share_longer_than_hours"] = long_
rep.append(f"\nShare of views longer than N hours: {long_}")
cat_dur = lab.groupby("tv_show_category").duration.describe()[["count", "25%", "50%", "75%"]]
cat_dur.to_csv(OUT / "labeled_duration_by_category.csv")
table(cat_dur)

fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
bins = np.logspace(np.log10(60), np.log10(log.duraton.max()), 60)
axes[0].hist(log.duraton, bins=bins, color=SERIES[0])
axes[0].set_xscale("log"); axes[0].set_title("View interval length (duraton)")
axes[0].set_xlabel("seconds (log scale)"); axes[0].set_ylabel("views")
for x, lbl in [(300, "5 min"), (3600, "1 h"), (6 * 3600, "6 h")]:
    axes[0].axvline(x, color="#52514e", lw=0.8, ls="--"); axes[0].text(x, axes[0].get_ylim()[1] * .95, f" {lbl}", fontsize=8)
b2 = np.logspace(np.log10(60), np.log10(40000), 50)
axes[1].hist(sched.loc[sched.tv_show_id == 0, "duration"].clip(60, 40000), bins=b2, density=True, histtype="step", lw=2,
             color=SERIES[1], label="unlabeled (tv_show_id = 0)")
axes[1].hist(lab.duration.clip(60, 40000), bins=b2, density=True, histtype="step", lw=2, color=SERIES[0], label="labeled")
axes[1].set_xscale("log"); axes[1].set_title("Airing duration in schedule (normalized)")
axes[1].set_xlabel("seconds (log scale)"); axes[1].set_ylabel("density"); axes[1].legend()
save(fig, OUT / "durations.png")

# ------------------------------------------------------------- test users
section("Test users")
tu = set(tpl.user_id)
per = {"weeks 11-20": log[log.start_time < config.W21], "weeks 21-30": log[log.start_time >= config.W21]}
cov = {"template_users": len(tu), "unique_in_template": int(tpl.user_id.nunique()),
       "in_log_any": len(tu & set(log.user_id))}
for k, v in per.items():
    cov[f"in_log_{k}"] = len(tu & set(v.user_id))
summary["test_users"] = cov
act = log.groupby("user_id").agg(views=("duraton", "size"), hours=("duraton", lambda x: x.sum() / 3600),
                                 active_days=("start_time", lambda x: x.dt.date.nunique()))
act["is_test"] = act.index.isin(tu)
act_cmp = act.groupby("is_test").describe().T.unstack(level=1)
act_cmp = act.groupby("is_test")[["views", "hours", "active_days"]].median()
act_cmp["n_users"] = act.groupby("is_test").size()
act_cmp.to_csv(OUT / "test_vs_other_activity.csv")
rep.append("```\n" + json.dumps(cov, indent=1) + "\n```\n\nMedian activity (weeks 11-30), test vs other users:\n")
table(act_cmp.round(1))

# ------------------------------------------------------------- template
section("Submission template")
ok_format = tpl.tv_show_id.str.split().map(len).eq(5).all()
summary["template"] = {"rows": len(tpl), "columns": list(tpl.columns), "example": tpl.iloc[0].astype(str).to_dict(),
                       "each_row_has_5_space_separated_ids": bool(ok_format)}
rep.append("```\n" + json.dumps(summary["template"], indent=1, ensure_ascii=False) + "\n```")

json.dump(summary, open(OUT / "eda_summary.json", "w"), indent=1, ensure_ascii=False, default=str)
(OUT / "eda_report.md").write_text("# EDA report\n" + "\n".join(rep), encoding="utf-8")
print("\n".join(rep))
