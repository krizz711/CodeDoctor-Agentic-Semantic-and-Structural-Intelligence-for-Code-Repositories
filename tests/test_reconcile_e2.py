import copy
import json
from pathlib import Path

import pytest

from scripts.reconcile_e2 import assess


@pytest.fixture
def real_run():
    return json.loads((Path(__file__).parents[1] / "experiments/C3_cycle/E2_runs/e2_A_run1_VOID.json").read_text())


def valid_negative(real_run):
    result = copy.deepcopy(real_run)
    for arm in ("T1", "T1b"):
        gate = result[f"gate_{arm}"]
        gate["REF_within_25pct_of_7.83s"] = True
        gate["raw_latencies"]["REF"] = [7.832] * 100
    return result


def test_recorded_void_cannot_be_adopted(real_run):
    r = assess(real_run, attempt=1)
    assert r["status"] == "void"
    assert "one documented repeat" in r["next_action"]
    assert not r["adoption"]


def test_second_void_stops(real_run):
    assert assess(real_run, attempt=2)["next_action"] == "stop E2; no more repeats"


def test_valid_negative_is_not_void(real_run):
    r = assess(valid_negative(real_run), attempt=2)
    assert r["status"] == "negative", r


def test_incomplete_does_not_approve_followup(real_run):
    result = valid_negative(real_run)
    del result["gate_T1b"]
    assert assess(result, attempt=2)["status"] == "incomplete"


def test_passing_gate_requires_existing_stage_b(real_run):
    result = valid_negative(real_run)
    gate = result["gate_T1"]
    gate.update(cos_vs_stored={"median": .999, "min": .99}, **{"pass": True})
    gate["efficiency"].update(ratio=.7, ci95=[.65, .75])
    result["chosen"] = "T1"
    del result["gate_T1b"]
    r = assess(result, attempt=2)
    assert r["status"] == "gate_passed", r
    assert not r["adoption"]


def test_false_control_flag_overrides_positive_gate(real_run):
    result = valid_negative(real_run)
    result["NEG_fails_as_required"] = False
    assert assess(result, attempt=2)["status"] == "void"


def test_ref_mean_is_checked_even_if_flag_says_pass(real_run):
    result = valid_negative(real_run)
    result["gate_T1"]["raw_latencies"]["REF"] = [20.] * 100
    assert assess(result, attempt=2)["status"] == "void"


def test_no_third_attempt(real_run):
    with pytest.raises(ValueError):
        assess(real_run, attempt=3)


def test_empty_result_is_incomplete():
    assert assess({}, attempt=2)["status"] == "incomplete"
