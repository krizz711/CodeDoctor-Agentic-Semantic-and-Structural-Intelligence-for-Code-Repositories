#!/usr/bin/env bash
# Second recording queue: waits for the first, then deepening, hubness pools, UniXcoder.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1
run() { local name=$1; shift; echo "[$(date +%T)] $name"; "$@" > "experiments/logs/$name.log" 2>&1; echo "[$(date +%T)] $name done"; }
run deepen_06b_dev  $PY scripts/deepen_outcomes.py --store data/outcomes/f2llm06b_all_dev.jsonl.gz --to 50 --workers 6
run rec_06b_hub_dev $PY scripts/record_outcomes.py --model f2llm06b --split dev --workers 6 \
    --hubness-bank artifacts/ext_cc/emb_f2llm06b_queries.npy --hubness-beta 0.3 \
    --out data/outcomes/f2llm06b_all_dev_hubcc.jsonl.gz
run rec_unix_dev    $PY scripts/record_outcomes.py --model unixcoder --split dev --workers 6
run deepen_unix_dev $PY scripts/deepen_outcomes.py --store data/outcomes/unixcoder_all_dev.jsonl.gz --to 50 --workers 6
echo "[$(date +%T)] cpu queue 2 complete"
