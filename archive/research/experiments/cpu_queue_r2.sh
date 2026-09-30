#!/usr/bin/env bash
# R2: re-record the 0.6B stores under the same code and conditions as the 1.7B ones.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1
run() { local name=$1; shift; echo "[$(date +%T)] $name"; "$@" > "experiments/logs/$name.log" 2>&1; echo "[$(date +%T)] $name done"; }
run rec_06b_dev_r2     $PY scripts/record_outcomes.py --model f2llm06b --split dev  --workers 6 --out data/outcomes/f2llm06b_all_dev_r2.jsonl.gz
run rec_06b_tune_r2    $PY scripts/record_outcomes.py --model f2llm06b --split tune --workers 6 --out data/outcomes/f2llm06b_all_tune_r2.jsonl.gz
run deepen_06b_dev_r2  $PY scripts/deepen_outcomes.py --store data/outcomes/f2llm06b_all_dev_r2.jsonl.gz  --to 50 --workers 6
run deepen_06b_tune_r2 $PY scripts/deepen_outcomes.py --store data/outcomes/f2llm06b_all_tune_r2.jsonl.gz --to 50 --workers 6
echo "[$(date +%T)] cpu queue r2 complete"
