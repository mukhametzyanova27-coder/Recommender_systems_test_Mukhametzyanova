"""Step 9. Results table (method, MAP@5 on validation, hyperparameters, run
time), library versions, environment parameters and seeds.

Outputs: outputs/environment.json, outputs/RESULTS.md
"""
import json
import os
import platform
import sys
from importlib import metadata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from sweettv import config

pkgs = ["numpy", "pandas", "scipy", "scikit-learn", "implicit", "kagglehub", "pyarrow", "matplotlib", "tabulate", "threadpoolctl"]
env = {
    "python": sys.version, "platform": platform.platform(), "machine": platform.machine(), "cpu_count": os.cpu_count(),
    "packages": {p: metadata.version(p) for p in pkgs},
    "seeds": {"global (config.SEED)": config.SEED, "implicit ALS random_state": config.SEED, "bootstrap numpy rng": config.SEED},
    "threads": {"OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS", "1 (set inside scripts 03/04)"),
                "implicit ALS num_threads": 8},
    "data_source": f"kagglehub.competition_download('{config.COMPETITION}') or $SWEETTV_DATA_DIR",
}
try:
    from threadpoolctl import threadpool_info
    env["blas"] = [{k: i.get(k) for k in ("internal_api", "version", "num_threads")} for i in threadpool_info()]
except Exception as e:  # pragma: no cover
    env["blas"] = str(e)
json.dump(env, open(config.out("environment.json"), "w"), indent=1)

V = config.OUTPUT_DIR / "validation"
best = json.load(open(V / "best_config.json"))
res = pd.read_csv(V / "results.csv")
final = json.load(open(config.OUTPUT_DIR / "final" / "final_report.json"))
tab = res[["method", "map5_test", "se_test", "map5_other", "map5_all", "diff_vs_pop_test", "diff_vs_pop_test_ci95",
           "fit_sec", "predict_sec_test_users", "model_filled_5", "fallback_only", "params"]].copy()
tab = tab.rename(columns={"map5_test": "MAP@5 test users (tuning)", "se_test": "SE", "map5_other": "MAP@5 other users (confirm)",
                          "map5_all": "MAP@5 all users", "diff_vs_pop_test": "diff vs pop", "diff_vs_pop_test_ci95": "95% CI",
                          "fit_sec": "fit, s", "predict_sec_test_users": "predict, s"})
md = ["# Results\n",
      f"Validation: train weeks 11-20, holdout weeks 21-30; truth = shows watched by rule `{best['holdout_rule']}`. "
      f"Test users = template users with non-empty holdout ({best['eval_users_test']}), "
      f"other users = remaining holdout users ({best['eval_users_other']}), never used for choices.\n",
      tab.round(4).to_markdown(index=False),
      "\n\nHybrid vs best single method (paired bootstrap):\n",
      pd.read_csv(V / "G_hybrid_vs_best_single.csv").round(4).to_markdown(index=False),
      f"\n\nFinal strategy: **{best['final_strategy']}** (retrained on weeks 11-30).\n",
      "Chosen configuration:\n\n```json\n" + json.dumps(best, indent=1, default=str) + "\n```\n",
      "Final run (weeks 11-30 -> 31-42):\n\n```json\n" + json.dumps(final, indent=1) + "\n```\n",
      "Environment:\n\n```json\n" + json.dumps(env, indent=1) + "\n```\n"]
(config.OUTPUT_DIR / "RESULTS.md").write_text("\n".join(md), encoding="utf-8")
print("\n".join(md))
