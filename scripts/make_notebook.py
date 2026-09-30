"""Generate a minimal notebook that runs the submission evaluation on a hosted machine.

    python scripts/make_notebook.py --out running.ipynb

The notebook is a thin runner around the documented commands: it unpacks the pipeline
source (embedded, so a private repository needs no clone), installs the pinned
requirements, builds the offline document cache (a GPU is allowed for that step by the
FAQ), runs ``scripts/evaluate_mteb.py`` with CPU query inference, and validates the
result. It contains no evaluation logic of its own; everything runs through the same
entrypoint a judge would run. The recorded, multi-stage version of the same commands is
``scripts/submission_run.py`` (see ``scripts/build_submission_kernel.py``).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.packaging import build_payload  # noqa: E402


def code(*lines: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in lines],
    }


def markdown(*lines: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": [line + "\n" for line in lines]}


def build_notebook(payload: str) -> dict:
    return {
        "cells": [
            markdown(
                "# CodeDoctor: AppsRetrieval evaluation (submission configuration)",
                "",
                "Runs `scripts/evaluate_mteb.py`, the official MTEB `AppsRetrieval` evaluation, with",
                "`config/submission.yaml`: F2LLM-v2-1.7B, **queries encoded on the CPU**, and the",
                "accepted execution-verification settings. The pipeline source is embedded below,",
                "generated from the repository by `scripts/make_notebook.py`.",
                "",
                "Document vectors are built once, offline (the FAQ allows a GPU for pre-processing),",
                "and the evaluation refuses to run if any document is missing from that cache.",
                "Encoding 3,765 queries with a 1.7B model on a CPU takes many hours; `--query-store`",
                "checkpoints each vector so an interrupted run resumes instead of restarting.",
                "",
                "The evaluation **executes Python that came from the dataset** in a sandboxed",
                "subprocess (a resource limiter, not a security boundary: no network or filesystem",
                "isolation). Run it in a disposable environment.",
            ),
            code(
                "# 1. Environment",
                "import os, platform, psutil, torch",
                "print(platform.platform())",
                "print('cores', psutil.cpu_count(logical=False), '/', os.cpu_count(),",
                "      ' RAM', round(psutil.virtual_memory().total / 1e9, 1), 'GB')",
                "print('torch', torch.__version__, 'cuda', torch.cuda.is_available())",
            ),
            code(
                "# 2. Unpack the pipeline (generated from the repository)",
                f'PAYLOAD = "{payload}"',
                "",
                "import base64, io, sys, tarfile",
                "",
                "WORK = os.environ.get('CODEDOCTOR_WORK', '/content/codedoctor')",
                "os.makedirs(WORK, exist_ok=True)",
                "with tarfile.open(fileobj=io.BytesIO(base64.b64decode(PAYLOAD)), mode='r:gz') as tar:",
                "    tar.extractall(WORK)",
                "sys.path.insert(0, WORK)",
                "os.chdir(WORK)",
                "print(sorted(os.listdir(WORK)))",
            ),
            code(
                "# 3. Pinned requirements (mteb decides the result schema and the dataset revision)",
                "!pip install -q -r requirements-submission.txt",
            ),
            markdown(
                "## Offline document cache",
                "",
                "Encodes the 8,765 corpus documents once and records the model, revision, dtype and",
                "checksums in `documents/manifest.json`. Use a GPU if there is one (about ten minutes on a",
                "T4); on a CPU this step alone takes hours.",
            ),
            code(
                "DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'",
                "print('document cache on', DEVICE)",
                "!python scripts/prepare_document_cache.py --device $DEVICE --out documents",
            ),
            markdown(
                "## The evaluation",
                "",
                "Query inference is on the CPU. Results go to `results/` (an existing output is never",
                "overwritten without `--overwrite`).",
            ),
            code(
                "!python scripts/evaluate_mteb.py --device cpu \\",
                "    --corpus-cache documents/corpus.sqlite --require-complete-corpus-cache \\",
                "    --query-store results/queries.sqlite \\",
                "    --predictions results/predictions --output results/appsretrieval_results.json",
                "!python scripts/validate_submission.py results/appsretrieval_results.json",
            ),
            code(
                "# Download the artifacts (Colab); on Kaggle they are already under /kaggle/working",
                "try:",
                "    from google.colab import files  # noqa: F401",
                "    for name in ('results/appsretrieval_results.json', 'results/submission_predictions.csv',",
                "                 'results/submission_metrics.csv', 'results/execution_audit.json',",
                "                 'documents/manifest.json'):",
                "        if os.path.exists(name):",
                "            files.download(name)",
                "except ImportError:",
                "    print('not on Colab: collect the files under', os.path.abspath('results'))",
            ),
        ],
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": []},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "running.ipynb")
    args = parser.parse_args()

    payload, included = build_payload()
    print(f"Packed {len(included)} paths -> {len(payload) / 1024:.0f} KB base64")
    args.out.write_text(json.dumps(build_notebook(payload), indent=1), encoding="utf-8")
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
