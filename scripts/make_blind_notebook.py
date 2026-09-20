"""Generate the Kaggle notebook that runs the BLIND generalisation suite on GPU.

The screening submission answers one question: how well does the pipeline do on
CoIR AppsRetrieval. It cannot answer the question a reviewer actually asks --
"did you build a general code-intelligence system, or did you tune against that
one benchmark?" -- because the benchmark is the thing in question.

This notebook answers it by running the *same shipped pipeline*, unchanged, on
retrieval tasks it was never tuned on, and reporting the numbers whatever they
are. It exists to be attached as proof of work, so it prints the frozen recipe
it is running under and writes one result JSON per task.

    python scripts/make_blind_notebook.py --out build/blind_notebook
    kaggle kernels push -p build/blind_notebook

Why a GPU notebook rather than a local run: encoding is the whole cost. On a
6-thread CPU the 0.6B encoder manages 0.35 texts/s, so CosQA's 21,104 texts
take roughly 17 hours. MTEB's own published evaluation_time for this encoder on
a comparable corpus is 75 seconds on an accelerator. The accuracy tier is worse
than slow locally -- the 1.7B weights are 3.4 GB and the machine this was
developed on has 7.7 GB of RAM.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.packaging import PAYLOAD_PATHS, build_payload  # noqa: E402

DEFAULT_SLUG = "codedoctor-blind-generalisation"

#: Tasks the pipeline was never tuned on, smallest corpus first so a notebook
#: that runs out of time still produces something. ``queries + corpus`` is the
#: number of texts that must be encoded.
BLIND_TASKS = [
    ("CodeTransOceanDL", 996, "code-to-code: find the equivalent program"),
    ("CodeTransOceanContest", 1229, "code-to-code, competition sources"),
    ("CosQA", 21104, "real natural-language web queries against Python functions"),
    ("StackOverflowQA", 21925, "Stack Overflow question against its answer post"),
]

#: Paths the default payload filter drops but this notebook needs.
EXTRA_PATHS = ("app/agent", "scripts/blind_eval.py")


def md(*lines: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": _src(lines)}


def code(*lines: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": _src(lines),
    }


def _src(lines: tuple[str, ...]) -> list[str]:
    body = "\n".join(lines)
    parts = body.split("\n")
    return [line + "\n" for line in parts[:-1]] + [parts[-1]]


def build_cells(payload: str, included: list[str], tasks: list[tuple[str, int, str]]) -> list[dict]:
    task_rows = "\n".join(
        "| `{0}` | {1:,} | {2} |".format(name, texts, blurb) for name, texts, blurb in tasks
    )
    task_literal = json.dumps([name for name, _, _ in tasks])

    return [
        md(
            "# CodeDoctor — blind generalisation suite",
            "",
            "Samsung PRISM GenAI Hackathon 2026 · Theme 01: Agentic Code Intelligence",
            "",
            "The submitted score is **NDCG@10 = 0.96543** on CoIR AppsRetrieval. This",
            "notebook does not reproduce that number — a separate notebook does. This one",
            "runs the *same pipeline, unchanged* against retrieval tasks it was never",
            "tuned on, and reports whatever comes out.",
            "",
            "| task | texts to encode | what it asks |",
            "|---|---|---|",
            task_rows,
            "",
            "None of these were used during development. No weights were fitted on them,",
            "no threshold was chosen against them, and the qrels are read only by the",
            "evaluator.",
        ),
        md(
            "## Read this before trusting any number below",
            "",
            "`mteb.evaluate` defaults to `cache=ResultCache(~/.cache/mteb)` with",
            "`overwrite_strategy='only-missing'`. The pipeline advertises the **upstream**",
            "model meta — `codefuse-ai/F2LLM-v2-*` at a pinned revision — so on a machine",
            "that has pulled MTEB's public results repository the evaluator will match a",
            "*published leaderboard result* and return it without running a single forward",
            "pass.",
            "",
            "It is a silent failure. The run finishes in ~11 seconds instead of ~47",
            "minutes, writes a well-formed result JSON, and reports the base encoder's",
            "score as though it were the pipeline's. We hit exactly this and nearly",
            "published it.",
            "",
            "`scripts/blind_eval.py` passes `cache=None, overwrite_strategy='always'`, and",
            "`tests/test_result_cache_disabled.py` fails if anyone removes it. **If a task",
            "below finishes in seconds, the number is not ours — check the cache.**",
        ),
        code(
            "# 1. Hardware",
            "import torch",
            "",
            "print('torch', torch.__version__, '| cuda:', torch.cuda.is_available())",
            "if torch.cuda.is_available():",
            "    major, minor = torch.cuda.get_device_capability(0)",
            "    total = torch.cuda.get_device_properties(0).total_memory / 1e9",
            "    print(f'{torch.cuda.get_device_name(0)}  compute {major}.{minor}  {total:.1f} GB')",
            "    print('native bfloat16:', major >= 8)",
            "else:",
            "    print('NO GPU. This notebook will take many hours on CPU -- stop and',",
            "          'switch the accelerator on.')",
        ),
        code(
            "# 2. Unpack the pipeline (embedded, so this notebook has no external deps)",
            "import base64, io, os, sys, tarfile",
            "",
            "WORK = '/kaggle/working/codedoctor'",
            "os.makedirs(WORK, exist_ok=True)",
            "PAYLOAD = '''" + payload + "'''",
            "with tarfile.open(fileobj=io.BytesIO(base64.b64decode(PAYLOAD)), mode='r:gz') as tar:",
            "    tar.extractall(WORK)",
            "if WORK not in sys.path:",
            "    sys.path.insert(0, WORK)",
            "os.chdir(WORK)",
            "print('unpacked:')",
            "for line in " + json.dumps(included) + ":",
            "    print('  ', line)",
        ),
        code(
            "# 3. Evaluation dependencies. mteb is pinned: it decides the result schema,",
            "#    the retrieval evaluator and the dataset revisions.",
            "!pip -q install 'mteb==2.21.0' 'sentence-transformers>=5.0,<7' 'datasets>=3.0' 2>&1 | tail -3",
        ),
        code(
            "# 4. The recipe this run uses, printed from the frozen config rather than",
            "#    described. Four of these decide the score.",
            "import yaml",
            "cfg = yaml.safe_load(open('config/frozen.yaml', encoding='utf-8'))",
            "enc = cfg['encoder']",
            "for key in ('production_model', 'revision', 'query_prefix', 'document_prefix',",
            "            'add_eos_token', 'max_seq_length', 'normalize_embeddings'):",
            "    print(f'{key:22s} {enc.get(key)!r}')",
            "print()",
            "print('interactive tier   ', cfg['interactive_encoder']['production_model'])",
        ),
        md(
            "## The run",
            "",
            "Each task runs through `scripts/blind_eval.py`, which imports the submission",
            "path's own `build_mteb_encoder`, `get_pipeline_config` and `get_tier_model`",
            "rather than reimplementing them. The only thing that differs between this and",
            "the screening run is the dataset handed to `mteb.evaluate`.",
            "",
            "Each task also reports **execution-eligible queries** before evaluating: how",
            "many carry worked examples that `parse_worked_examples` can read. That is a",
            "hard ceiling on what execution verification can contribute, and it is a",
            "property of the dataset, not of the wording. AppsRetrieval is 3676/3765",
            "(97.6%). Expect 0% on most of these.",
        ),
        code(
            "# 5. Interactive tier (0.6B) across every blind task.",
            "import subprocess, time",
            "",
            "TASKS = " + task_literal,
            "results = {}",
            "for task in TASKS:",
            "    print('=' * 72)",
            "    started = time.time()",
            "    proc = subprocess.run(",
            "        [sys.executable, '-u', 'scripts/blind_eval.py', '--task', task,",
            "         '--tier', 'interactive', '--device', 'auto', '--batch-size', '32'],",
            "        capture_output=True, text=True,",
            "    )",
            "    elapsed = time.time() - started",
            "    tail = [l for l in proc.stdout.splitlines() if l.strip()][-14:]",
            "    print('\\n'.join(tail))",
            "    if proc.returncode != 0:",
            "        print('FAILED\\n', proc.stderr[-1500:])",
            "    results[task] = elapsed",
            "    print(f'-> {task} finished in {elapsed:.1f}s')",
        ),
        code(
            "# 6. Accuracy tier (1.7B) -- the encoder the submission used. Run it on the",
            "#    two small tasks so the comparison is like-for-like with the screening",
            "#    encoder rather than only with the interactive one.",
            "for task in TASKS[:2]:",
            "    print('=' * 72)",
            "    proc = subprocess.run(",
            "        [sys.executable, '-u', 'scripts/blind_eval.py', '--task', task,",
            "         '--tier', 'accuracy', '--device', 'auto', '--batch-size', '16',",
            "         '--output', f'data/blind/{task}.accuracy.json'],",
            "        capture_output=True, text=True,",
            "    )",
            "    tail = [l for l in proc.stdout.splitlines() if l.strip()][-10:]",
            "    print('\\n'.join(tail))",
            "    if proc.returncode != 0:",
            "        print('FAILED\\n', proc.stderr[-1500:])",
        ),
        code(
            "# 7. Summary. Everything here is read back off disk, so the table cannot",
            "#    disagree with the artifacts.",
            "import glob, json",
            "",
            "rows = []",
            "for path in sorted(glob.glob('data/blind/*.json')):",
            "    payload = json.load(open(path, encoding='utf-8'))",
            "    scores = payload.get('scores', {}).get('test', [{}])[0]",
            "    meta = payload.get('codedoctor_blind_run', {})",
            "    rows.append({",
            "        'task': payload.get('task_name'),",
            "        'tier': meta.get('tier'),",
            "        'ndcg@10': scores.get('ndcg_at_10'),",
            "        'mrr@10': round(scores.get('mrr_at_10', 0), 5),",
            "        'recall@10': scores.get('recall_at_10'),",
            "        'exec-eligible': f\"{meta.get('execution_eligible_queries')}/{meta.get('total_queries')}\",",
            "        'seconds': meta.get('wall_clock_seconds'),",
            "    })",
            "",
            "try:",
            "    import pandas as pd",
            "    display(pd.DataFrame(rows))",
            "except ImportError:",
            "    print(json.dumps(rows, indent=2))",
            "",
            "print()",
            "print('A task that finished in single-digit seconds did NOT run the pipeline.')",
        ),
        md(
            "## How to read the result",
            "",
            "Expect these numbers to be **much lower than 0.96543**, and that is the",
            "honest outcome rather than a failure. Three reasons, in order of size:",
            "",
            "1. **Execution verification cannot fire.** It carried the screening score from",
            "   0.93756 to 0.96543, and it needs worked examples in the query. Cell 5",
            "   prints the eligible count per task; where it reads 0, the pipeline has",
            "   reduced to its dense pass and there is nothing of this project's",
            "   contribution left in the number.",
            "2. **The reranker ships as the identity configuration.** It did not clear its",
            "   significance gate (+0.0048, 95% CI spanning zero), so it contributes",
            "   nothing here either, by design.",
            "3. **These tasks are harder in absolute terms.** AppsRetrieval has exactly one",
            "   gold per query in a corpus of 8,765 near-unique programs. CosQA and",
            "   StackOverflowQA are noisier and more ambiguous.",
            "",
            "What the run *does* establish: the dense pass with the frozen recipe is not",
            "APPS-specific, it transfers to other code-retrieval tasks, and the pipeline",
            "does not collapse or error on data shaped nothing like its benchmark. Where a",
            "pass cannot contribute, it is skipped and reported as skipped rather than",
            "quietly credited.",
        ),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "build" / "blind_notebook")
    parser.add_argument("--slug", default=DEFAULT_SLUG)
    parser.add_argument("--username", default="krizz711")
    args = parser.parse_args()

    payload, included = build_payload(PAYLOAD_PATHS, extra=EXTRA_PATHS)
    missing = [p for p in EXTRA_PATHS if p not in included]
    if missing:
        raise SystemExit("payload is missing required paths: {0}".format(missing))

    notebook = {
        "cells": build_cells(payload, included, BLIND_TASKS),
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    args.out.mkdir(parents=True, exist_ok=True)
    notebook_path = args.out / "codedoctor-blind.ipynb"
    notebook_path.write_text(json.dumps(notebook, indent=1), encoding="utf-8")

    metadata = {
        "id": "{0}/{1}".format(args.username, args.slug),
        "title": "CodeDoctor - blind generalisation suite",
        "code_file": notebook_path.name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
    }
    (args.out / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    size_mb = notebook_path.stat().st_size / 1e6
    print("wrote {0}  ({1:.2f} MB)".format(notebook_path, size_mb))
    print("wrote {0}".format(args.out / "kernel-metadata.json"))
    print("payload carries {0} paths".format(len(included)))
    print()
    print("push with:  kaggle kernels push -p {0}".format(args.out))
    print("it is created PRIVATE (is_private: true); publish it yourself when ready.")


if __name__ == "__main__":
    main()
