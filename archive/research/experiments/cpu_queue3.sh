#!/usr/bin/env bash
# Replications on TUNE: deepening and hubness x execution. Runs after queue 2.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1
until grep -q "cpu queue 2 complete" experiments/logs_cpu_queue2.txt; do sleep 20; done
run() { local name=$1; shift; echo "[$(date +%T)] $name"; "$@" > "experiments/logs/$name.log" 2>&1; echo "[$(date +%T)] $name done"; }
run deepen_06b_tune  $PY scripts/deepen_outcomes.py --store data/outcomes/f2llm06b_all_tune.jsonl.gz --to 50 --workers 6
run rec_06b_hub_tune $PY scripts/record_outcomes.py --model f2llm06b --split tune --workers 6 \
    --hubness-bank artifacts/ext_cc/emb_f2llm06b_queries.npy --hubness-beta 0.3 \
    --out data/outcomes/f2llm06b_all_tune_hubcc.jsonl.gz
echo "[$(date +%T)] cpu queue 3 complete"
