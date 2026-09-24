#!/usr/bin/env bash
# R1: re-measure the v2 components with the shipped 1.7B encoder (prereg: experiments/R1_17b/prereg.md).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1
run() { local name=$1; shift; echo "[$(date +%T)] $name"; "$@" > "experiments/logs/$name.log" 2>&1; echo "[$(date +%T)] $name done"; }
run rec_17b_dev     $PY scripts/record_outcomes.py --model f2llm17b --split dev  --workers 6
run rec_17b_tune    $PY scripts/record_outcomes.py --model f2llm17b --split tune --workers 6
run deepen_17b_dev  $PY scripts/deepen_outcomes.py --store data/outcomes/f2llm17b_all_dev.jsonl.gz  --to 50 --workers 6
run deepen_17b_tune $PY scripts/deepen_outcomes.py --store data/outcomes/f2llm17b_all_tune.jsonl.gz --to 50 --workers 6
echo "[$(date +%T)] cpu queue 17b complete"
