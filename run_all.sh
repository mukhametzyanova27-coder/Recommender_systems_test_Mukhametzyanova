#!/usr/bin/env bash
# Full pipeline. Data: $SWEETTV_DATA_DIR or kagglehub download (needs Kaggle credentials).
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-python}
$PY scripts/01_eda.py
$PY scripts/02_join.py
$PY scripts/03_validate.py
$PY scripts/04_final.py
$PY scripts/05_examples.py
$PY scripts/06_report.py
$PY scripts/07_analysis.py
