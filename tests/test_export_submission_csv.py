"""Reject mislabeled predictions and preserve the evaluator's tie ordering."""

import csv
import importlib.util
import json

import pytest

from scripts.export_submission_csv import (
    _load_predictions, export_from_predictions, validate_predictions, verify_prediction_metrics,
)


def test_prediction_identity_and_finite_scores_are_checked():
    good = {"q1": {"d1": 1.0, "d2": 0.5}}
    validate_predictions(good, query_ids={"q1"}, corpus_ids={"d1", "d2"}, top_k=2)
    for bad in ({"q2": good["q1"]}, {"q1": {"unknown": 1.0, "d2": .5}},
                {"q1": {"d1": float("nan"), "d2": .5}}, {"q1": {"d1": True, "d2": .5}},
                {"q1": {"d1": 1.0}}):
        with pytest.raises(ValueError):
            validate_predictions(bad, query_ids={"q1"}, corpus_ids={"d1", "d2"}, top_k=2)


def test_ambiguous_prediction_files_are_rejected(tmp_path):
    for name in ("one.json", "two.json"):
        (tmp_path / name).write_text(json.dumps({"q1": {"d1": 1.0}}))
    with pytest.raises(ValueError, match="exactly one"):
        _load_predictions(tmp_path, strict=True)


def test_csv_ties_follow_document_id_descending_without_score_rounding(tmp_path):
    predictions = tmp_path / "predictions"
    predictions.mkdir()
    source = {"q1": {"d1": .9, "d2": .9, "d0": .9000000001}}
    (predictions / "AppsRetrieval_predictions.json").write_text(json.dumps(source))
    out = tmp_path / "output"
    export_from_predictions(predictions, {"scores": {"test": [{"ndcg_at_10": 1.0, "mrr_at_10": 1.0}]}}, out)
    with (out / "submission_predictions.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert [row["corpus_id"] for row in rows] == ["d0", "d2", "d1"]
    assert all(float(row["score"]) == source["q1"][row["corpus_id"]] for row in rows)
    assert rows[0]["score"] != rows[1]["score"]


@pytest.mark.skipif(importlib.util.find_spec("mteb") is None or importlib.util.find_spec("pytrec_eval") is None,
                    reason="optional official metric stack not installed")
def test_required_metrics_are_recomputed_from_raw_predictions():
    predictions = {"q1": {"d1": 0.5, "d2": 1.0}, "q2": {"d2": 1.0, "d1": 0.1}}
    qrels = {"q1": {"d1": 1}, "q2": {"d2": 1}}
    scores = {"ndcg_at_10": .81546, "mrr_at_10": .75}
    assert verify_prediction_metrics(predictions, qrels, scores) == scores
    with pytest.raises(ValueError, match="disagree"):
        verify_prediction_metrics(predictions, qrels, {**scores, "ndcg_at_10": .99})
