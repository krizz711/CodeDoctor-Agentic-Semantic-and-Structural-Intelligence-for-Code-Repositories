import json

from scripts.validate_submission import validate


def test_validate_local_result(tmp_path):
    path = tmp_path / "result.json"
    path.write_text(json.dumps({"metrics": {"mrr": 0.8, "ndcg_at_10": 0.7}, "predictions": []}))
    assert validate(path) == []


def test_validate_rejects_invalid_metric(tmp_path):
    path = tmp_path / "result.json"
    path.write_text(json.dumps({"metrics": {"mrr": 1.5}}))
    assert "metrics.mrr must be a number between 0 and 1" in validate(path)
