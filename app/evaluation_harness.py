from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import random
from typing import Callable, Mapping, Sequence
import numpy as np


class LeakageFirewallViolation(RuntimeError):
    """Raised when test split is accessed while development firewall is active."""
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


_APPS_DATASET_CACHE: tuple[list[dict], list[dict], dict[str, str]] | None = None


def _load_apps_data() -> tuple[list[dict], list[dict], dict[str, str]]:
    global _APPS_DATASET_CACHE
    if _APPS_DATASET_CACHE is not None:
        return _APPS_DATASET_CACHE

    try:
        from datasets import load_dataset

        qrels = load_dataset("CoIR-Retrieval/apps", "default")
        queries_ds = load_dataset("CoIR-Retrieval/apps", "queries", split="queries")
        corpus_ds = load_dataset("CoIR-Retrieval/apps", "corpus", split="corpus")

        query_lookup = {row["_id"]: row["text"] for row in queries_ds}
        corpus_lookup = {row["_id"]: row["text"] for row in corpus_ds}

        train_rows = []
        for r in qrels["train"]:
            qid = r["query-id"]
            cid = r["corpus-id"]
            train_rows.append({
                "query_id": qid,
                "query": query_lookup.get(qid, ""),
                "gold_id": cid,
                "gold_doc": corpus_lookup.get(cid, ""),
            })

        test_rows = []
        for r in qrels["test"]:
            qid = r["query-id"]
            cid = r["corpus-id"]
            test_rows.append({
                "query_id": qid,
                "query": query_lookup.get(qid, ""),
                "gold_id": cid,
                "gold_doc": corpus_lookup.get(cid, ""),
            })

        _APPS_DATASET_CACHE = (train_rows, test_rows, corpus_lookup)
        return _APPS_DATASET_CACHE
    except Exception:
        # Fallback to deterministic synthetic set if Hugging Face data unavailable
        synth_train = [
            {"query_id": f"q_{i}", "query": f"python function {i}", "gold_id": f"doc_{i}", "gold_doc": f"def doc_{i}(): pass"}
            for i in range(100)
        ]
        synth_test = [
            {"query_id": f"test_q_{i}", "query": f"test query {i}", "gold_id": f"test_doc_{i}", "gold_doc": f"def test_doc_{i}(): pass"}
            for i in range(50)
        ]
        synth_corpus = {r["gold_id"]: r["gold_doc"] for r in synth_train + synth_test}
        return (synth_train, synth_test, synth_corpus)


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
