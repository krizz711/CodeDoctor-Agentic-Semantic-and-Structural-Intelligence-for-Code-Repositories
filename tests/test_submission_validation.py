import json
from scripts.validate_submission import validate
from scripts.evaluate import main as evaluate_main


def test_validate_mteb_task_result_shape(tmp_path):
    path = tmp_path / "mteb_result.json"
    mteb_payload = {
        "dataset_revision": "1.0",
        "mteb_version": "2.21.0",
        "task_name": "AppsRetrieval",
        "scores": {
            "test": [
                {
                    "ndcg_at_10": 0.9045,
                    "mrr_at_10": 0.8812,
                }
            ]
        },
    }
    path.write_text(json.dumps(mteb_payload))
    assert validate(path) == []


def test_validate_rejects_all_zero_poisoned_artifact(tmp_path):
    """Section 3: Must reject all-zero poisoned scores."""
    path = tmp_path / "poisoned.json"
    path.write_text(json.dumps({
        "scores": {
            "test": [
                {
                    "ndcg_at_10": 0.0,
                    "mrr_at_10": 0.0,
                }
            ]
        }
    }))
    errors = validate(path)
    assert len(errors) == 2
    assert any("strictly positive" in e for e in errors)


def test_validate_rejects_flat_wrong_keys(tmp_path):
    """Section 3: Must reject flat or malformed structures."""
    path = tmp_path / "flat.json"
    path.write_text(json.dumps({"ndcg": 0.9}))
    errors = validate(path)
    assert len(errors) > 0
    assert any("Missing or invalid 'scores'" in e for e in errors)


def test_local_evaluation_output_is_validator_compatible(tmp_path, monkeypatch):
    output = tmp_path / "sample_results.json"
    monkeypatch.setattr("sys.argv", [
        "evaluate.py",
        "scripts/sample_eval.jsonl",
        "--output",
        str(output),
    ])

    evaluate_main()

    assert validate(output) == []
