#!/usr/bin/env bash
# Offline encodes for this round, run back to back on the local GPU.
# Every step is offline pre-processing (the FAQ allows a GPU); none touches the
# APPS test split.  Each step logs to experiments/logs/<step>.log and a failure
# stops the queue so a broken artifact is never silently used downstream.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python
mkdir -p experiments/logs
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1
BUDGET="--attention-budget 16000000 --max-batch 16"

run() { local name=$1; shift; echo "[$(date +%T)] $name"; "$@" > "experiments/logs/$name.log" 2>&1; echo "[$(date +%T)] $name done"; }

# external CodeContests set: reuse APPS document vectors for the 8,765 APPS distractors
run ext_queries  $PY scripts/encode_jsonl.py artifacts/ext_cc/texts_queries.jsonl --mode query \
    --out artifacts/ext_cc/emb_f2llm06b_queries.npy --device cuda $BUDGET
run ext_corpus   $PY scripts/encode_jsonl.py artifacts/ext_cc/texts_corpus.jsonl --mode document \
    --out artifacts/ext_cc/emb_f2llm06b_corpus.npy --device cuda $BUDGET \
    --reuse artifacts/apps/texts_corpus.jsonl artifacts/apps/emb_f2llm06b_corpus.npy

# version benchmark: queries are APPS train statements (reuse), corpus reuses unchanged APPS docs
run ver_queries  $PY scripts/encode_jsonl.py artifacts/versions_apps/texts_queries.jsonl --mode query \
    --out artifacts/versions_apps/emb_f2llm06b_queries.npy --device cuda $BUDGET \
    --reuse artifacts/apps/texts_queries_train.jsonl artifacts/apps/emb_f2llm06b_qtrain.npy
run ver_corpus   $PY scripts/encode_jsonl.py artifacts/versions_apps/texts_corpus.jsonl --mode document \
    --out artifacts/versions_apps/emb_f2llm06b_corpus.npy --device cuda $BUDGET \
    --reuse artifacts/apps/texts_corpus.jsonl artifacts/apps/emb_f2llm06b_corpus.npy

# cascade first tier: F2LLM-v2-160M, same recipe, pinned revision
run f2llm160m    $PY scripts/encode_apps.py --model-name codefuse-ai/F2LLM-v2-160M \
    --revision 4ffe22c31406fc321b65a05b15563d38275ee51f --tag f2llm160m --device cuda \
    --splits corpus,qtrain $BUDGET

# contamination control: an encoder that predates CoIR/APPS
run unixcoder    $PY scripts/encode_baseline.py --model microsoft/unixcoder-base --tag unixcoder --device cuda
echo "[$(date +%T)] queue complete"
