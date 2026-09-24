#!/usr/bin/env bash
# R1 addendum: external replication and versions at 1.7B. Runs after cpu_queue_17b.sh.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1
until grep -q "cpu queue 17b complete" experiments/logs_cpu_queue_17b.txt; do sleep 20; done
run() { local name=$1; shift; echo "[$(date +%T)] $name"; "$@" > "experiments/logs/$name.log" 2>&1; echo "[$(date +%T)] $name done"; }
run rec_17b_ext_dev    $PY scripts/record_outcomes.py --external artifacts/ext_cc --model f2llm17b --workers 6 \
    --out data/outcomes/f2llm17b_extcc_dev.jsonl.gz
run deepen_17b_ext_dev $PY scripts/deepen_outcomes.py --store data/outcomes/f2llm17b_extcc_dev.jsonl.gz \
    --corpus artifacts/ext_cc/texts_corpus.jsonl --to 50 --workers 6
run rec_17b_versions   $PY scripts/record_outcomes.py --external artifacts/versions_apps --model f2llm17b --workers 6 \
    --out data/outcomes/f2llm17b_versions.jsonl.gz
echo "[$(date +%T)] cpu queue 17b ext complete"
