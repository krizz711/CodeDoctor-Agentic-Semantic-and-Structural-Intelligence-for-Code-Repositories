"""The demo response keeps the per-result verification fields its UI reads.

web/dist/index.html renders ``verification.examples`` and ``verification.note``.
The shared verifier replaced the demo's own loop; these tests pin that the
records it produces still carry both, so the UI never shows "undefined".
"""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from app.retrieval.hybrid import Document, RankedDocument
from app.submission import execution_config
from app.workbench import Workbench

QUERY = "Print the sum.\n\n-----Example-----\nInput\n2 3\n\nOutput\n5\n"
GOOD = "a,b=map(int,input().split()); print(a+b)"
WRONG = "a,b=map(int,input().split()); print(a-b)"
CRASH = "raise ValueError('boom')"


def doc(code, name):
    return Document(document_id=name, file_path="solve.py", content=code, name=name, type="FUNCTION")


def verify(codes, **overrides):
    config = replace(execution_config(), workers=1, **overrides)
    hits = [RankedDocument(doc(code, f"c{i}"), 1.0 - i / 100, ("dense",)) for i, code in enumerate(codes)]
    built = SimpleNamespace(target=SimpleNamespace(language="Python"))
    return Workbench()._verify(QUERY, hits, built, enabled=True, config=config)


def test_verified_and_failed_records_carry_examples_and_a_reason():
    result = verify([WRONG, CRASH, GOOD])
    states = [r["state"] for r in result["per_result"]]
    assert states == ["failed", "failed", "verified"]
    for record in result["per_result"]:
        assert set(record) >= {"state", "examples", "note"}
        assert record["examples"] == 1
    assert result["per_result"][0]["note"] == "output mismatch"
    assert "ValueError" in result["per_result"][1]["note"]
    assert result["per_result"][2]["note"] == ""


def test_candidates_below_the_depth_say_so_instead_of_staying_blank():
    result = verify([WRONG, GOOD, WRONG], max_candidates=1, deepen_to=0)
    record = result["per_result"][2]
    assert record["state"] == "not_run"
    assert record["examples"] == 1 and record["note"] == "below the configured execution depth"


def test_a_statement_without_examples_reports_not_run_without_a_crash():
    config = replace(execution_config(), workers=1)
    hits = [RankedDocument(doc(GOOD, "c0"), 1.0, ("dense",))]
    built = SimpleNamespace(target=SimpleNamespace(language="Python"))
    result = Workbench()._verify("Explain how sums work.", hits, built, enabled=True, config=config)
    assert result["ran"] is False
    assert result["per_result"] == [{"state": "not_run"}]
    assert "worked examples" in result["reason"]


def _ready_bench(tmp_path):
    import time

    root = tmp_path / "proj"
    root.mkdir()
    (root / "m.py").write_text("def total(a, b):\n    return a + b\n", encoding="utf-8")
    bench = Workbench()
    key = bench.open_folder(str(root))["key"]
    bench.start_build(key, dense=False)
    deadline = time.time() + 60
    while time.time() < deadline and bench.state()["phase"] not in ("ready", "error"):
        time.sleep(0.05)
    assert bench.state()["phase"] == "ready"
    return bench, key


def test_search_says_which_configuration_and_model_it_ran_under(tmp_path, monkeypatch):
    bench, key = _ready_bench(tmp_path)
    monkeypatch.delenv("CODEDOCTOR_TIER", raising=False)
    result = bench.search(key, "total", verify=False)
    ident = result["submission"]
    assert ident["configuration"] == "config/submission.yaml"
    assert ident["tier"] == "accuracy" and ident["model"].endswith("F2LLM-v2-1.7B")
    assert ident["execution"]["blank_line_fallback"] is True and ident["execution"]["deepen_to"] == 50
    assert {"rank", "verification", "retrieval_score"} <= set(result["results"][0])  # existing fields kept
    assert "encoder" in result and "timings" in result and "isolation" in result


def test_the_smaller_model_is_opt_in_and_identified(tmp_path, monkeypatch):
    bench, key = _ready_bench(tmp_path)
    monkeypatch.setenv("CODEDOCTOR_TIER", "interactive")
    ident = bench.search(key, "total", verify=False)["submission"]
    assert ident["tier"] == "interactive" and ident["model"].endswith("F2LLM-v2-0.6B")


def test_an_unknown_tier_fails_instead_of_picking_a_model(monkeypatch):
    monkeypatch.setenv("CODEDOCTOR_TIER", "fast")
    with pytest.raises(ValueError, match="CODEDOCTOR_TIER"):
        Workbench._tier()
