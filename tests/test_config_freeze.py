from pathlib import Path
import pytest
import yaml

from app.retrieval.encode import CodeEncoder, FROZEN_CONFIG_PATH, DEFAULT_CONFIG_PATH
from scripts.evaluate_mteb import get_default_model, get_default_config_path


def test_submission_reads_frozen_config_not_live_config(tmp_path, monkeypatch):
    """
    Plan Remediation §3.9:
    Asserts that the submission / evaluation path reads from config/frozen.yaml
    and is provably insulated from post-freeze mutations to config/retrieval.yaml.
    """
    assert FROZEN_CONFIG_PATH.exists(), "config/frozen.yaml must exist as an I2 freeze artifact"

    frozen = yaml.safe_load(FROZEN_CONFIG_PATH.read_text(encoding="utf-8"))
    live_backup = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))

    expected_production_model = frozen["encoder"]["production_model"]

    # Mutate retrieval.yaml
    try:
        mutated_live = dict(live_backup)
        mutated_live["encoder"]["production_model"] = "some-other-model-nobody-approved"
        DEFAULT_CONFIG_PATH.write_text(yaml.safe_dump(mutated_live), encoding="utf-8")

        # Check default config path selected is frozen
        config_path = get_default_config_path()
        assert config_path == FROZEN_CONFIG_PATH

        # Check default model selected is frozen production model
        model_used = get_default_model()
        assert model_used == expected_production_model

        # The submission tier reads the frozen production model, not the live one
        encoder = CodeEncoder(tier="accuracy")
        assert encoder.model_name == expected_production_model
        assert encoder.config_path == FROZEN_CONFIG_PATH

        # ... and the interactive tier reads the frozen interactive model,
        # which is likewise unaffected by the mutation above.
        interactive = CodeEncoder(tier="interactive")
        assert interactive.model_name == frozen["interactive_encoder"]["production_model"]
        assert interactive.config_path == FROZEN_CONFIG_PATH
    finally:
        # Restore retrieval.yaml
        DEFAULT_CONFIG_PATH.write_text(yaml.safe_dump(live_backup), encoding="utf-8")
