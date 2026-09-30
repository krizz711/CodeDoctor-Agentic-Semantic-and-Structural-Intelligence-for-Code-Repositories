"""Package the evaluation pipeline into a self-contained payload.

The Colab notebook embeds the current source so a private repository needs no clone.
GPU document preprocessing and CPU query retrieval use the same evaluator as the
recorded Kaggle stages. The Kaggle builder separately packages exact committed blobs.
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
    "requirements.txt",
    "requirements-submission.txt",
    "scripts/__init__.py",
    "scripts/evaluate_mteb.py",
    "scripts/export_submission_csv.py",
    "scripts/prepare_document_cache.py",
    # The hosted notebook calls this after the run.  It was missing until
    # 2026-09-19, which meant the Colab notebook's validation cell died with
    # "can't open file scripts/validate_submission.py" -- the run was fine, the
    # check on it never executed.
    "scripts/validate_submission.py",
)


TEXT_SUFFIXES = {".py", ".yaml", ".yml", ".json", ".txt", ".md", ".cfg", ".toml"}


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
    a ``.npz`` hub profile.  Text files are added with LF line endings, so the
    payload is the same whichever OS builds it (a Windows checkout with autocrlf
    would otherwise ship CRLF sources).
    """
    buffer = io.BytesIO()
    included: list[str] = []
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for relative in (*paths, *extra):
            source = root / relative
            if not source.exists():
                continue
            files = [source] if source.is_file() else sorted(p for p in source.rglob("*") if p.is_file())
            for path in files:
                name = path.relative_to(root).as_posix()
                if _keep(tarfile.TarInfo(name)) is None:
                    continue
                data = path.read_bytes()
                if path.suffix in TEXT_SUFFIXES:
                    data = data.replace(b"\r\n", b"\n")
                info = tarfile.TarInfo(name)
                info.size, info.mode = len(data), 0o644
                tar.addfile(info, io.BytesIO(data))
            included.append(relative)
    return base64.b64encode(buffer.getvalue()).decode("ascii"), included
