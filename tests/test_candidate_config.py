"""The v2 candidate differs from the frozen submission only where evidence says so."""

from __future__ import annotations

from pathlib import Path

import yaml

from scripts.evaluate_mteb import get_pipeline_config

ROOT = Path(__file__).resolve().parent.parent
FROZEN = ROOT / "config" / "frozen.yaml"
CANDIDATE = ROOT / "config" / "candidate_v2.yaml"


def test_the_encoder_recipe_is_identical():
    frozen = yaml.safe_load(FROZEN.read_text(encoding="utf-8"))
    candidate = yaml.safe_load(CANDIDATE.read_text(encoding="utf-8"))
    assert candidate["encoder"] == frozen["encoder"]
    assert candidate["interactive_encoder"] == frozen["interactive_encoder"]


def test_the_frozen_submission_keeps_every_new_switch_off():
    config = get_pipeline_config(FROZEN)
    assert config["execution_call_style"] is False
    assert config["execution_extended_stdin"] is False
    assert config["execution_judge_rules"] is False
    assert config["execution_timeout_mode"] == "wall"
    assert config["execution_max_dense_gap"] is None
    assert config["execution_deepen_to"] == 0
    assert config["execution_skip_interactive"] is False
    assert config["hubness_bank"] is None


def test_the_candidate_turns_on_only_what_passed():
    config = get_pipeline_config(CANDIDATE)
    assert config["execution_call_style"] is True
    # The 0.20 dense gate was switched off after it excluded golds ~20x more
    # often on CodeContests than on APPS (experiments/STRESS_TEST.md, F4).
    assert config["execution_max_dense_gap"] is None
    assert config["execution_timeout_mode"] == "cpu"
    assert config["execution_deepen_to"] == 50
    assert config["execution_skip_interactive"] is True
    # measured and below the bar: stay off
    assert config["execution_judge_rules"] is False
    assert config["execution_extended_stdin"] is False
    # unchanged from the submission
    frozen = get_pipeline_config(FROZEN)
    for key in ("execution_rerank", "execution_candidates", "execution_timeout_seconds",
                "execution_numeric_tolerance"):
        assert config[key] == frozen[key], key
