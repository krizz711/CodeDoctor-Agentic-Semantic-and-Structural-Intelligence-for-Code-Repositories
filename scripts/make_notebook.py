"""Generate a Colab notebook that runs the evaluation on a hosted GPU.

The pipeline is embedded as a base64 tarball built from the working tree at
generation time — the same packer the Kaggle kernel builder uses — so the
notebook can never ship code that has drifted from this repository.

    python scripts/make_notebook.py --out running.ipynb
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
                "# CodeDoctor — AppsRetrieval evaluation on a hosted GPU",
                "",
                "Runs the two-pass retrieval pipeline through `mteb.evaluate` and writes the",
                "submission artifact. The pipeline source is embedded below, generated from",
                "the repository by `scripts/make_notebook.py` — there is no stale bundle to",
                "drift out of sync.",
                "",
                "The *solution* is CPU-only; a GPU is used here purely to encode 17,530 texts",
                "once. Everything downstream runs on cached vectors.",
            ),
            code(
                "# 1. Environment",
                "import torch",
                "",
                "print('torch', torch.__version__, 'cuda', torch.cuda.is_available())",
                "if torch.cuda.is_available():",
                "    name = torch.cuda.get_device_name(0)",
                "    major = torch.cuda.get_device_capability(0)[0]",
                "    mem = torch.cuda.get_device_properties(0).total_memory / 1e9",
                "    print(f'{name}  compute {major}.x  {mem:.1f} GB')",
                "    # Turing (T4, compute 7.5) has no native bfloat16: torch emulates it and",
                "    # the encode slows by roughly 10x. select_dtype handles this for us.",
                "else:",
                "    print('No GPU. This will work but the encode takes hours.')",
            ),
            code(
                "# 2. Unpack the pipeline (generated from the repo working tree)",
                f'PAYLOAD = "{payload}"',
                "",
                "import base64, io, os, sys, tarfile",
                "",
                "WORK = '/content/codedoctor'",
                "os.makedirs(WORK, exist_ok=True)",
                "with tarfile.open(fileobj=io.BytesIO(base64.b64decode(PAYLOAD)), mode='r:gz') as tar:",
                "    tar.extractall(WORK)",
                "sys.path.insert(0, WORK)",
                "os.chdir(WORK)",
                "print(sorted(os.listdir(WORK)))",
            ),
            code(
                "# 3. Dependencies (mteb pinned: it decides the result-JSON schema)",
                "!pip install -q -U 'mteb==2.21.0' 'sentence-transformers>=5.0' "
                "'transformers>=4.57' datasets pyyaml",
            ),
            markdown(
                "## Encode once",
                "",
                "Dumps texts, qrels and embeddings to `artifacts/apps`. Reranker tuning then",
                "runs on those cached vectors in seconds instead of hours.",
            ),
            code(
                "!python scripts/encode_apps.py --tier interactive --device auto --tag f2llm06b",
                "!python scripts/encode_apps.py --tier accuracy    --device auto --tag f2llm17b",
            ),
            markdown(
                "## Fit the second pass on TRAIN, report on held-out DEV",
                "",
                "The dev split is seeded-random and filtered to the query family the test",
                "split is actually made of — see `app/pipeline/splits.py` for why a",
                "contiguous half-split of this corpus is not exchangeable.",
            ),
            code("!python scripts/tune_reranker.py --model f2llm17b --depth 50"),
            markdown("## Stage-by-stage ablation, each gain gated on a paired bootstrap"),
            code(
                "!python scripts/dev_eval.py --model f2llm17b",
                "!python scripts/dev_eval.py --model f2llm17b --execution",
            ),
            markdown(
                "## The submission artifact",
                "",
                "Produced by `mteb.evaluate` itself — no score is assembled by hand.",
            ),
            code(
                "!python scripts/evaluate_mteb.py --device auto --predictions build/predictions",
                "!python scripts/validate_submission.py appsretrieval_results.json",
            ),
            code(
                "# Download the artifacts",
                "from google.colab import files  # noqa: F401",
                "for name in ('appsretrieval_results.json', 'submission_predictions.csv',",
                "             'submission_metrics.csv', 'config/reranker.json'):",
                "    if os.path.exists(name):",
                "        files.download(name)",
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
