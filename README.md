# sweet.tv — TV program recommender

A solution to the Kaggle course competition `sweettv-tv-program-recommender`. For each of the 1,260 test users
the model predicts 5 TV programs (`tv_show_id`) that the user will watch in weeks 31–42. The metric is MAP@5.

* File to upload to Kaggle: **`submission.csv`** in the repository root. A copy is in `outputs/final/submission.csv`.
* Full report with decisions, results and analysis: [`REPORT.pdf`](REPORT.pdf) (source: [`REPORT.md`](REPORT.md)).
* Kaggle score of `submission.csv` (late submission, after the deadline): public 0.42619, private 0.40540.

The final strategy is a hybrid of ALS (implicit) and content-based TF-IDF. On the temporal validation
(training on weeks 11–20, holdout on weeks 21–30, 1,244 test users) its MAP@5 is 0.6597, versus 0.5863 for the
popularity baseline.

## Project structure

```
.
├── README.md
├── REPORT.md                 report (Markdown source)
├── REPORT.pdf                report (PDF built from REPORT.md)
├── requirements.txt          pinned library versions
├── run_all.sh                runs the whole pipeline
├── submission.csv            final submission (copy of outputs/final/submission.csv)
├── sweettv/                  package with the logic
│   ├── config.py             paths (no hardcoded data paths), week windows, SEED = 42
│   ├── data.py               loading of log and schedule (+ parquet cache), program catalog
│   ├── channel_check.py      statistical check of the vsetv_id ↔ channel_id mapping
│   ├── join.py               interval join, "watched" rules, interaction table
│   ├── recs.py               shared rules: candidate pool, repeats, fallback, completion to 5 ids
│   ├── models.py             Content-based (TF-IDF), ALS (implicit), score/rank fusion
│   ├── metrics.py            AP@K / MAP@K
│   ├── evaluation.py         evaluation on fixed user sets
│   └── plots.py              figure styling
├── scripts/
│   ├── 01_eda.py             EDA                                       → outputs/eda/
│   ├── 02_join.py            join and threshold comparison             → outputs/join/
│   ├── 03_validate.py        validation, hyperparameter tuning, hybrid → outputs/validation/
│   ├── 04_final.py           retraining on weeks 11–30, submission     → outputs/final/, ./submission.csv
│   ├── 05_examples.py        example users                             → outputs/examples/
│   ├── 06_report.py          summary table and environment             → outputs/RESULTS.md, outputs/environment.json
│   └── 07_analysis.py        analysis for the report (familiar/new, activity, coverage) → outputs/analysis/
└── outputs/                  all run results (outputs/cache/ is not stored in git)
```

## Data

The data is not part of the repository. Data paths are not hardcoded; there are two ways to provide the data.

1. **kagglehub (default).** If `SWEETTV_DATA_DIR` is not set, the code calls
   ```python
   import kagglehub
   path = kagglehub.competition_download('sweettv-tv-program-recommender')
   ```
   This requires Kaggle credentials (for example, a token in `~/.kaggle/`) and accepted competition rules.
   The data is cached in `~/.cache/kagglehub/`.
2. **Local folder.** Download the competition files manually and point to the folder:
   ```bash
   export SWEETTV_DATA_DIR=/path/to/sweettv-data
   ```
   The folder must contain `dataset11-30.csv`, `export_arh_11-20-final.csv`, `export_arh_21-30-final.csv`,
   `export_arh_31-42-final.csv` and `submission.csv` (the template).

## How to run

Requires Python 3.10 (tested on 3.10.14, Linux x86_64).

```bash
git clone https://github.com/mukhametzyanova27-coder/Recommender_systems_test_Mukhametzyanova.git
cd Recommender_systems_test_Mukhametzyanova
python -m venv .venv
.venv/bin/pip install -r requirements.txt
PYTHON=.venv/bin/python ./run_all.sh          # steps 01–07 in order
```

The steps can also be run one at a time (`.venv/bin/python scripts/0N_*.py`), but strictly in order: each step
reads the results of the previous ones from `outputs/`. Intermediate tables are cached in `outputs/cache/`.
`SWEETTV_OUTPUT_DIR` moves all results to another folder, but the root `submission.csv` is always overwritten.

The longest step is `03_validate.py` (a grid of 960 + 108 + 15 ALS configurations). Timings and the
reproducibility check are in Section 10 of `REPORT.md`. Seeds are fixed (`config.SEED = 42`): a re-run produces
the same `submission.csv`.

## Where to find things

| What | Where |
|---|---|
| Submission for Kaggle | `submission.csv` (= `outputs/final/submission.csv`) |
| Submissions of individual methods, for comparison | `outputs/final/submission_{popularity,content,als,hybrid}.csv` |
| Submission format checks | `outputs/final/final_report.json` |
| MAP@5 table, hyperparameters, timings | `outputs/validation/results.csv`, `outputs/RESULTS.md` |
| Chosen configuration | `outputs/validation/best_config.json` |
| Tuning grids | `outputs/validation/{A..G}_*.csv` |
| EDA and channel check | `outputs/eda/eda_report.md`, `outputs/eda/channel_mapping_check.csv` |
| Distributions used to choose the threshold | `outputs/join/` |
| Example users | `outputs/examples/examples.md` |
| Analysis for the report | `outputs/analysis/analysis.md` |
| Library versions, seeds | `outputs/environment.json`, `requirements.txt` |
| Step timings and reproducibility check | `outputs/runtime_clean_run.txt` |
| Kaggle submission screenshot | `outputs/kaggle_submission.png` |

## Building the PDF report

`REPORT.pdf` was built from `REPORT.md` with pandoc 3.1.3 and XeLaTeX (run from the repository root so that the
figure paths resolve):

```bash
pandoc REPORT.md -o REPORT.pdf --pdf-engine=xelatex \
  -V mainfont="DejaVu Sans" -V monofont="DejaVu Sans Mono" \
  -V geometry:margin=1.5cm -V fontsize=10pt \
  -V colorlinks=true -V linkcolor=blue -V urlcolor=blue
```
