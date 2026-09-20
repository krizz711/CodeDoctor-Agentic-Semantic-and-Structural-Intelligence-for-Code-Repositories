from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Mapping, Sequence
import numpy as np


class LeakageFirewallViolation(RuntimeError):
    """Raised when test split is accessed while development firewall is active."""
    pass


class AppsDataUnavailable(RuntimeError):
    """Raised when neither the local APPS cache nor the HuggingFace copy loads.

    This is deliberately fatal.  An earlier version of this module answered the
    same situation with a hundred rows of ``python function {i}`` placeholder
    text, which meant every consumer -- ``dev_eval.py``, ``run_ablation.py``,
    ``measure_recall_ceiling.py`` -- silently reported metrics computed on
    synthetic data that looked entirely plausible.  A missing dataset is a
    setup problem and must read as one.
    """
    pass


@dataclass
class EvaluationMetrics:
    ndcg_at_10: float
    mrr_at_10: float
    recall_at_10: float
    recall_at_50: float
    recall_at_100: float
    recall_at_200: float

    def to_dict(self) -> dict:
        return {
            "ndcg_at_10": round(self.ndcg_at_10, 4),
            "mrr_at_10": round(self.mrr_at_10, 4),
            "recall_at_10": round(self.recall_at_10, 4),
            "recall_at_50": round(self.recall_at_50, 4),
            "recall_at_100": round(self.recall_at_100, 4),
            "recall_at_200": round(self.recall_at_200, 4),
        }


def compute_metrics_for_query(ranked_doc_ids: Sequence[str], gold_doc_id: str) -> dict[str, float]:
    """Computes exact NDCG@10, MRR, and Recall@K for a single-gold query."""
    # Find rank of gold document (1-indexed)
    rank = None
    for idx, doc_id in enumerate(ranked_doc_ids, start=1):
        if doc_id == gold_doc_id:
            rank = idx
            break

    mrr_10 = (1.0 / rank) if (rank is not None and rank <= 10) else 0.0
    ndcg_10 = (1.0 / math.log2(rank + 1)) if (rank is not None and rank <= 10) else 0.0

    return {
        "mrr_at_10": mrr_10,
        "ndcg_at_10": ndcg_10,
        "recall_at_10": 1.0 if (rank is not None and rank <= 10) else 0.0,
        "recall_at_50": 1.0 if (rank is not None and rank <= 50) else 0.0,
        "recall_at_100": 1.0 if (rank is not None and rank <= 100) else 0.0,
        "recall_at_200": 1.0 if (rank is not None and rank <= 200) else 0.0,
    }


def paired_bootstrap_test(
    scores_a: Sequence[float],
    scores_b: Sequence[float],
    n_resamples: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict[str, float | bool]:
    """
    Plan T5: Paired bootstrap over per-query NDCG.
    Produces a 95% confidence interval for diff (A - B).
    A component only ships if the confidence interval excludes zero.
    """
    assert len(scores_a) == len(scores_b), "Score lists must be paired and equal length"
    n = len(scores_a)
    diffs = np.array(scores_a) - np.array(scores_b)
    observed_mean_diff = float(np.mean(diffs))

    rng = np.random.default_rng(seed)
    boot_diffs = np.zeros(n_resamples, dtype=np.float64)

    for i in range(n_resamples):
        indices = rng.integers(0, n, size=n)
        boot_diffs[i] = np.mean(diffs[indices])

    lower_pct = (alpha / 2.0) * 100.0
    upper_pct = (1.0 - alpha / 2.0) * 100.0

    ci_lower = float(np.percentile(boot_diffs, lower_pct))
    ci_upper = float(np.percentile(boot_diffs, upper_pct))

    # Statistically significant if entire CI is strictly positive (or negative)
    significant = (ci_lower > 0) or (ci_upper < 0)

    return {
        "mean_diff": observed_mean_diff,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "significant": significant,
        "noise_band_width": ci_upper - ci_lower,
    }


#: Frozen texts and qrels written by ``scripts/encode_apps.py``.
APPS_CACHE_DIR = Path(__file__).resolve().parent.parent / "artifacts" / "apps"

_APPS_DATASET_CACHE: tuple[list[dict], list[dict], dict[str, str]] | None = None


def _rows_from_local_cache(
    split: str, texts: dict[str, str], corpus: dict[str, str]
) -> list[dict]:
    qrels = json.loads((APPS_CACHE_DIR / f"qrels_{split}.json").read_text(encoding="utf-8"))
    rows = []
    for query_id, relevant in qrels.items():
        for doc_id, relevance in relevant.items():
            if relevance > 0:
                rows.append({
                    "query_id": query_id,
                    "query": texts.get(query_id, ""),
                    "gold_id": doc_id,
                    "gold_doc": corpus.get(doc_id, ""),
                })
    return rows


def _read_texts(path: Path) -> dict[str, str]:
    out = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                record = json.loads(line)
                out[record["id"]] = record["text"]
    return out


def _load_from_local_cache() -> tuple[list[dict], list[dict], dict[str, str]] | None:
    """The frozen copy written by ``scripts/encode_apps.py``.

    Same bytes the encoders and the MTEB run see, so the harness cannot drift
    from the rest of the pipeline.  Returns ``None`` when the cache is absent.
    """
    required = [
        APPS_CACHE_DIR / name
        for name in (
            "texts_corpus.jsonl",
            "texts_queries_train.jsonl",
            "texts_queries_test.jsonl",
            "qrels_train.json",
            "qrels_test.json",
        )
    ]
    if not all(path.exists() for path in required):
        return None

    corpus = _read_texts(APPS_CACHE_DIR / "texts_corpus.jsonl")
    train = _rows_from_local_cache(
        "train", _read_texts(APPS_CACHE_DIR / "texts_queries_train.jsonl"), corpus
    )
    test = _rows_from_local_cache(
        "test", _read_texts(APPS_CACHE_DIR / "texts_queries_test.jsonl"), corpus
    )
    return train, test, corpus


def _load_from_huggingface() -> tuple[list[dict], list[dict], dict[str, str]]:
    from datasets import load_dataset

    qrels = load_dataset("CoIR-Retrieval/apps", "default")
    queries_ds = load_dataset("CoIR-Retrieval/apps", "queries", split="queries")
    corpus_ds = load_dataset("CoIR-Retrieval/apps", "corpus", split="corpus")

    query_lookup = {row["_id"]: row["text"] for row in queries_ds}
    corpus_lookup = {row["_id"]: row["text"] for row in corpus_ds}

    def rows(split: str) -> list[dict]:
        return [
            {
                "query_id": r["query-id"],
                "query": query_lookup.get(r["query-id"], ""),
                "gold_id": r["corpus-id"],
                "gold_doc": corpus_lookup.get(r["corpus-id"], ""),
            }
            for r in qrels[split]
        ]

    return rows("train"), rows("test"), corpus_lookup


def _load_apps_data() -> tuple[list[dict], list[dict], dict[str, str]]:
    """Real CoIR AppsRetrieval rows: local cache first, then HuggingFace.

    Raises :class:`AppsDataUnavailable` if neither source works.  It never
    substitutes placeholder rows -- see that exception's docstring.
    """
    global _APPS_DATASET_CACHE
    if _APPS_DATASET_CACHE is not None:
        return _APPS_DATASET_CACHE

    cached = _load_from_local_cache()
    if cached is not None:
        _APPS_DATASET_CACHE = cached
        return _APPS_DATASET_CACHE

    try:
        _APPS_DATASET_CACHE = _load_from_huggingface()
    except Exception as exc:
        raise AppsDataUnavailable(
            "Could not load the CoIR AppsRetrieval split "
            f"({type(exc).__name__}: {exc}).\n"
            "Either populate the local cache with\n"
            "    python scripts/encode_apps.py --tier accuracy --device auto "
            "--tag f2llm17b\n"
            f"which writes texts and qrels to {APPS_CACHE_DIR}, or install the\n"
            "evaluation extras so the HuggingFace copy can be downloaded:\n"
            "    pip install -r requirements-eval.txt"
        ) from exc
    return _APPS_DATASET_CACHE


AUDIT_LOG_PATH = Path(__file__).resolve().parent.parent / "data" / "test_split_audit_log.json"


def get_persistent_audit_log(path: Path = AUDIT_LOG_PATH) -> list[dict]:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return []
    return []


def record_persistent_audit(
    reason: str,
    row_count: int,
    caller: str = "DevelopmentHarness.get_test_split",
    path: Path = AUDIT_LOG_PATH,
    extra: dict | None = None,
) -> dict:
    from datetime import datetime, timezone
    path.parent.mkdir(parents=True, exist_ok=True)
    logs = get_persistent_audit_log(path)
    entry = {
        "access_index": len(logs) + 1,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "reason": reason,
        "caller": caller,
        "split": "test",
        "rows": row_count,
        "status": "COMPLETED",
    }
    if extra:
        entry.update(extra)
    logs.append(entry)
    path.write_text(json.dumps(logs, indent=2), encoding="utf-8")
    return entry


class DevelopmentHarness:
    """
    Development harness for training split exploration with strict leakage firewall.
    """

    def __init__(self, firewall_enabled: bool = True, audit_log_path: Path | None = None) -> None:
        self.firewall_enabled = firewall_enabled
        self.audit_log_path = audit_log_path
        self.test_split_access_count = 0
        self.tuning_runs_logged = 0

    def set_firewall(self, enabled: bool) -> None:
        self.firewall_enabled = enabled

    def get_train_split(self) -> list[dict]:
        """Provides access to the 5,000-row train split."""
        self.tuning_runs_logged += 1
        train_rows, _, _ = _load_apps_data()
        return list(train_rows)

    def get_test_split(self, authorize_reason: str | None = None) -> list[dict]:
        """
        Plan T2: Leakage firewall.
        If firewall is ON, this method raises LeakageFirewallViolation.
        Enforces maximum 2 lifetime accesses when audit_log_path is configured.
        """
        if self.firewall_enabled:
            raise LeakageFirewallViolation(
                "LEAKAGE FIREWALL ACTIVE: Access to test split is forbidden during development. "
                "Tune on train split exclusively."
            )
        if authorize_reason not in ("calibration", "final_run"):
            raise PermissionError(
                f"Unauthorized test split access: reason '{authorize_reason}' not permitted. "
                "Allowed reasons: 'calibration', 'final_run'."
            )

        if self.audit_log_path is not None:
            existing_logs = get_persistent_audit_log(self.audit_log_path)
            if len(existing_logs) >= 2:
                raise PermissionError(
                    f"TEST SPLIT BUDGET EXHAUSTED: Maximum of 2 test split accesses allowed per Plan T2. "
                    f"Current count = {len(existing_logs)}. Cannot spend additional test touches."
                )

        self.test_split_access_count += 1
        _, test_rows, _ = _load_apps_data()

        if self.audit_log_path is not None:
            record_persistent_audit(
                reason=authorize_reason,
                row_count=len(test_rows),
                caller="DevelopmentHarness.get_test_split",
                path=self.audit_log_path,
            )

        return list(test_rows)

    @classmethod
    def persistent_access_count(cls, path: Path = AUDIT_LOG_PATH) -> int:
        return len(get_persistent_audit_log(path))

    def get_corpus(self) -> dict[str, str]:
        _, _, corpus_lookup = _load_apps_data()
        return dict(corpus_lookup)

    def evaluate_predictions(
        self, query_predictions: Mapping[str, Sequence[str]], query_golds: Mapping[str, str]
    ) -> EvaluationMetrics:
        """Computes aggregate NDCG@10, MRR@10, and Recall@10,50,100,200."""
        mrr_sum = 0.0
        ndcg_sum = 0.0
        r10_sum = 0.0
        r50_sum = 0.0
        r100_sum = 0.0
        r200_sum = 0.0

        count = max(1, len(query_golds))
        for qid, gold_id in query_golds.items():
            ranked = query_predictions.get(qid, [])
            metrics = compute_metrics_for_query(ranked, gold_id)
            mrr_sum += metrics["mrr_at_10"]
            ndcg_sum += metrics["ndcg_at_10"]
            r10_sum += metrics["recall_at_10"]
            r50_sum += metrics["recall_at_50"]
            r100_sum += metrics["recall_at_100"]
            r200_sum += metrics["recall_at_200"]

        return EvaluationMetrics(
            ndcg_at_10=ndcg_sum / count,
            mrr_at_10=mrr_sum / count,
            recall_at_10=r10_sum / count,
            recall_at_50=r50_sum / count,
            recall_at_100=r100_sum / count,
            recall_at_200=r200_sum / count,
        )
