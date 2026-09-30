"""Package the evaluation pipeline into a self-contained payload.

The screening run wants a GPU, and the free ones live in hosted notebooks.
Rather than publishing the repository to get the code there — or, worse,
committing a stale zip that silently diverges from the source — the payload is
built from the working tree at generation time.

One packer, used by both ``scripts/build_eval_kernel.py`` (Kaggle) and
``scripts/make_notebook.py`` (Colab), so the two can never carry different code.
"""

from __future__ import annotations

import base64
import io
import tarfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

#: Everything the evaluation path imports, and nothing else.
PAYLOAD_PATHS: tuple[str, ...] = (
    "app/__init__.py",
    "app/pipeline",
    "app/retrieval",
    "app/evaluation.py",
    "app/submission.py",
    "app/indexing",
    "app/evaluation_harness.py",
    "app/packaging.py",
    "config/frozen.yaml",
    "config/submission.yaml",
    "config/retrieval.yaml",
    "config/reranker.json",
    "scripts/__init__.py",
    "scripts/evaluate_mteb.py",
    "scripts/export_submission_csv.py",
    "scripts/encode_apps.py",
    "scripts/tune_reranker.py",
    "scripts/dev_eval.py",
    # Both hosted notebooks call these after the run.  They were missing until
    # 2026-09-19, which meant the Colab notebook's validation cell died with
    # "can't open file scripts/validate_submission.py" -- the run was fine, the
    # check on it never executed.
    "scripts/validate_submission.py",
    "scripts/audit_execution_rerank.py",
    # Gold labels for the post-run decomposition.  87 KB, and it keeps the
    # audit runnable in a hosted notebook that has no artifacts/ directory.
    "data/qrels_test.json",
)


def _keep(info: tarfile.TarInfo) -> tarfile.TarInfo | None:
    if "__pycache__" in info.name or info.name.endswith((".pyc", ".DS_Store")):
        return None
    return info


def build_payload(
    paths: tuple[str, ...] = PAYLOAD_PATHS,
    *,
    extra: tuple[str, ...] = (),
    root: Path = REPO_ROOT,
) -> tuple[str, list[str]]:
    """Return ``(base64 gzipped tar, included paths)`` built from the working tree.

    ``extra`` adds paths that the default filter would otherwise drop, such as
    a ``.npz`` hub profile.
    """
    buffer = io.BytesIO()
    included: list[str] = []
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for relative in (*paths, *extra):
            source = root / relative
            if not source.exists():
                continue
            tar.add(source, arcname=relative, filter=_keep)
            included.append(relative)
    return base64.b64encode(buffer.getvalue()).decode("ascii"), included
