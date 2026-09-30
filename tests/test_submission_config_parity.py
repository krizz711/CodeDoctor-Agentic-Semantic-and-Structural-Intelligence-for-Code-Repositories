"""config/submission.yaml is the accepted candidate, and nothing else changed.

The historical frozen configuration stays as it was (its result is the
provenance of the first release); the submission configuration differs from it
only by the switches that passed their pre-registered rules.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import yaml

from app.submission import execution_config, get_pipeline_config

ROOT = Path(__file__).resolve().parent.parent
FROZEN = ROOT / "config" / "frozen.yaml"
CANDIDATE = ROOT / "config" / "candidate_v2.yaml"
SUBMISSION = ROOT / "config" / "submission.yaml"

#: Only what passed a pre-registered decision (experiments/RESULTS.md, cycle 3).
FROZEN_SHA256 = "b0e068ff105bd8b5e053f49c29f79dd0655ae1ebbfde586b4975ed52f8c7375e"

ACCEPTED_ON = {
    "execution_call_style", "execution_deepen_to", "execution_skip_interactive",
    "execution_timeout_mode", "execution_blank_line_fallback", "execution_judge_like_io",
}


def load(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_the_encoder_recipe_is_the_published_one():
    frozen, submission = load(FROZEN), load(SUBMISSION)
    # The one documented addition: a CPU-only batch cap (batching is slower on CPU; it only changes padding).
    assert submission["encoder"].pop("cpu_max_batch") == 1
    assert submission["encoder"] == frozen["encoder"]
    assert submission["interactive_encoder"] == frozen["interactive_encoder"]
    assert submission["encoder"]["production_model"] == "codefuse-ai/F2LLM-v2-1.7B"
    assert submission["encoder"]["revision"] == "3766d46e7a68545ed6190c15330983f9b39ab718"


def test_the_submission_pipeline_equals_the_accepted_candidate():
    assert load(SUBMISSION)["pipeline"] == load(CANDIDATE)["pipeline"]


def test_only_accepted_switches_differ_from_the_historical_recipe():
    frozen, submission = get_pipeline_config(FROZEN), get_pipeline_config(SUBMISSION)
    changed = {key for key in frozen if frozen[key] != submission[key]}
    assert changed == ACCEPTED_ON, changed ^ ACCEPTED_ON


def test_measured_and_rejected_options_stay_off():
    config = get_pipeline_config(SUBMISSION)
    assert config["execution_max_dense_gap"] is None      # gate: excluded golds on CodeContests
    assert config["execution_judge_rules"] is False       # below the minimum effect
    assert config["execution_extended_stdin"] is False
    assert config["hubness_bank"] is None                 # did not replicate with execution


def test_execution_config_reaches_every_accepted_switch():
    built = execution_config()
    assert (built.call_style, built.blank_line_fallback, built.judge_like_io) == (True, True, True)
    assert (built.skip_interactive, built.deepen_to, built.timeout_mode) == (True, 50, "cpu")
    assert (built.max_candidates, built.timeout_seconds, built.numeric_tolerance) == (20, 2.0, 1e-6)


def test_the_historical_frozen_file_is_unchanged():
    """Provenance of appsretrieval_results.json: any edit must be a deliberate, reviewed one."""
    digest = hashlib.sha256(FROZEN.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    assert digest == FROZEN_SHA256, "config/frozen.yaml changed; it is the first release's provenance"


@pytest.mark.parametrize("bad", ["", "   \n", "pipeline: [unclosed", "{{ not yaml", "- a\n- b\n"])
def test_an_unreadable_or_empty_configuration_fails_visibly(tmp_path, bad):
    path = tmp_path / "broken.yaml"
    path.write_text(bad, encoding="utf-8")
    with pytest.raises(ValueError):
        get_pipeline_config(path)


def test_a_missing_configuration_fails_visibly(tmp_path):
    with pytest.raises(FileNotFoundError):
        get_pipeline_config(tmp_path / "absent.yaml")


def test_cpu_encoding_is_one_sequence_at_a_time_and_the_gpu_and_frozen_recipes_are_not_touched():
    from app.retrieval.encode import CodeEncoder

    assert CodeEncoder(device="cpu").max_batch == 1
    assert CodeEncoder(device="cuda").max_batch == 64
    assert CodeEncoder(device="cpu", config_path=FROZEN).max_batch == 64
