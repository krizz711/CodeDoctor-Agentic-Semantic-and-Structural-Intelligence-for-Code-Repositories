#!/usr/bin/env bash
# Execution recordings for this round, sequential, 6 workers (~physical cores:
# more workers made wall-clock timeouts load-dependent, see experiments/README).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1
run() { local name=$1; shift; echo "[$(date +%T)] $name"; "$@" > "experiments/logs/$name.log" 2>&1; echo "[$(date +%T)] $name done"; }
run rec_ext_dev   $PY scripts/record_outcomes.py --external artifacts/ext_cc --model f2llm06b --workers 6 \
    --out data/outcomes/f2llm06b_extcc_dev.jsonl.gz
run rec_versions  $PY scripts/record_outcomes.py --external artifacts/versions_apps --model f2llm06b --workers 6 \
    --out data/outcomes/f2llm06b_versions.jsonl.gz
run rec_160m_dev  $PY scripts/record_outcomes.py --model f2llm160m --split dev --workers 6
run rec_06b_tune  $PY scripts/record_outcomes.py --model f2llm06b --split tune --workers 6
echo "[$(date +%T)] cpu queue complete"
