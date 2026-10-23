"""Step 7. Retrain on weeks 11-30 with the configuration chosen on validation
and write submission.csv (template order, exactly 5 unique ids per user).

Outputs: outputs/final/submission_<method>.csv, outputs/final/final_report.json,
         submission.csv (project root, the file to upload; = final strategy)
"""
import json
import os
import shutil
import sys
import time
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from sweettv import config, data
from sweettv.models import ALS, ContentBased, fuse
from sweettv.recs import Split, complete, topn_from_scores

OUT = config.out("final")
OUT.mkdir(parents=True, exist_ok=True)
best = json.load(open(config.OUTPUT_DIR / "validation" / "best_config.json"))
rep = best["allow_repeats"]

sched = data.load_schedule()
tpl = data.load_template()
ua = pd.read_parquet(config.CACHE_DIR / "ua_final_train.parquet")
split = Split("final", best["train_rule"], sched, ua)
catalog = data.items_catalog(sched[sched["start"] < config.W43])
users = tpl["user_id"].to_numpy()
fallback = list(split.popularity(best["pop_recent_weeks"]).index)

timing = {}
t = time.time()
cb = ContentBased(catalog, best["content"]["field_weights"], best["content"]["weighting"],
                  best["content"]["half_life_weeks"]).fit(split.inter, split.end_sec)
timing["content_fit_sec"] = time.time() - t
t = time.time()
als = ALS(**best["als"], seed=config.SEED).fit(split.inter, split.end_sec)
timing["als_fit_sec"] = time.time() - t
S_cb, S_als = cb.score(users, split.pool), als.score(users, split.pool)
scores = {
    "popularity": None,
    "content": S_cb,
    "als": S_als,
    "hybrid": fuse(S_als, S_cb, best["hybrid"]["w_als"], best["hybrid"]["fusion"]),
}

pool = set(split.pool.tolist())
report = {"train_rule": best["train_rule"], "allow_repeats": rep, "pool_size": len(pool), "timing": timing,
          "final_strategy": best["final_strategy"], "methods": {}}
for name, S in scores.items():
    ranked = {} if S is None else topn_from_scores(S, users, split.pool, split.seen, rep)
    recs = complete(ranked, users, fallback, split.seen, rep)
    sub = pd.DataFrame({"user_id": tpl["user_id"].astype(str),
                        "tv_show_id": [" ".join(str(int(i)) for i in recs[u]) for u in users]})
    sub.to_csv(OUT / f"submission_{name}.csv", index=False)
    n_model = np.array([min(len(ranked.get(u, [])), config.K) for u in users])
    report["methods"][name] = {"model_filled_5": int((n_model == config.K).sum()),
                               "model_filled_partly": int(((n_model > 0) & (n_model < config.K)).sum()),
                               "fallback_only": int((n_model == 0).sum()),
                               "distinct_ids_recommended": int(len({i for u in users for i in recs[u]}))}

# ---------------------------------------------------------------- format check
final_path = OUT / f"submission_{best['final_strategy']}.csv"
sub = pd.read_csv(final_path, dtype=str)
tpl_raw = pd.read_csv(config.data_dir() / config.SUBMISSION_TEMPLATE, dtype=str)
ids = sub["tv_show_id"].str.split()
checks = {
    "columns_match_template": list(sub.columns) == list(tpl_raw.columns),
    "rows": len(sub), "rows_match_template": len(sub) == len(tpl_raw),
    "user_order_matches_template": bool((sub["user_id"].values == tpl_raw["user_id"].values).all()),
    "every_row_5_ids": bool(ids.map(len).eq(5).all()),
    "every_row_unique_ids": bool(ids.map(lambda x: len(set(x)) == 5).all()),
    "all_ids_integer_nonzero": bool(ids.map(lambda x: all(i.isdigit() and int(i) != 0 for i in x)).all()),
    "all_ids_in_target_window_pool": bool(ids.map(lambda x: all(int(i) in pool for i in x)).all()),
    "no_missing_values": bool(sub.notna().all().all()),
}
report["format_checks"] = checks
assert all(v for k, v in checks.items() if k != "rows"), checks
shutil.copyfile(final_path, OUT / "submission.csv")
shutil.copyfile(final_path, config.PROJECT_ROOT / "submission.csv")
json.dump(report, open(OUT / "final_report.json", "w"), indent=1)
print(json.dumps(report, indent=1))
print(sub.head())
