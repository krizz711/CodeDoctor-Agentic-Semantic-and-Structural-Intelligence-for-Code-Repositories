"""The frozen config must reproduce the published F2LLM-v2 retrieval recipe.

A silently wrong prefix or a silently truncating sequence cap costs ~3 NDCG
points and leaves no error behind — the run just scores lower than the model
can.  These tests pin every value that decides the score, and where MTEB is
installed they pin it against MTEB's own model registry rather than a constant
copied by hand.
"""

from __future__ import annotations


import pytest
import yaml

from app.retrieval.encode import (
    APPS_INSTRUCTION,
    APPS_QUERY_PREFIX,
    F2LLM_INSTRUCTION_TEMPLATE,
    FROZEN_CONFIG_PATH,
    PUBLISHED_MAX_SEQ_LENGTH,
    CodeEncoder,
    plan_batches,
)

ACCURACY_MODEL = "codefuse-ai/F2LLM-v2-1.7B"
INTERACTIVE_MODEL = "codefuse-ai/F2LLM-v2-0.6B"


@pytest.fixture(scope="module")
def frozen() -> dict:
    return yaml.safe_load(FROZEN_CONFIG_PATH.read_text(encoding="utf-8"))


class TestFrozenConfig:
    def test_query_prefix_is_the_apps_task_instruction(self, frozen):
        assert frozen["encoder"]["query_prefix"] == APPS_QUERY_PREFIX
        assert APPS_QUERY_PREFIX == F2LLM_INSTRUCTION_TEMPLATE.format(
            instruction=APPS_INSTRUCTION
        )

    def test_documents_are_encoded_without_an_instruction(self, frozen):
        assert frozen["encoder"]["document_prefix"] == ""

    def test_eos_token_is_appended(self, frozen):
        # Pooling is last-token; without an EOS the pooled vector is whatever
        # token happened to land last.
        assert frozen["encoder"]["add_eos_token"] is True

    def test_sequence_cap_matches_the_published_run(self, frozen):
        for key in ("max_seq_length", "query_max_seq_length", "document_max_seq_length"):
            assert frozen["encoder"][key] == PUBLISHED_MAX_SEQ_LENGTH, key

    def test_embeddings_are_normalized_and_scored_by_cosine(self, frozen):
        assert frozen["encoder"]["normalize_embeddings"] is True
        assert frozen["encoder"]["similarity_metric"] == "cosine"

    def test_both_tiers_are_pinned_to_a_revision(self, frozen):
        assert frozen["encoder"]["production_model"] == ACCURACY_MODEL
        assert frozen["interactive_encoder"]["production_model"] == INTERACTIVE_MODEL
        for section in ("encoder", "interactive_encoder"):
            revision = frozen[section]["revision"]
            assert isinstance(revision, str) and len(revision) == 40, section

    def test_execution_rerank_is_on_and_its_limits_are_pinned(self, frozen):
        """It ships ON so a bare reproduce run matches the submitted number.

        It is the only pass that cleared the paired-bootstrap gate. The limits
        are pinned alongside it because they are what makes running
        dataset-authored code acceptable at all.
        """
        pipeline = frozen["pipeline"]
        assert pipeline["execution_rerank"] is True
        assert pipeline["execution_candidates"] == 10
        assert pipeline["execution_timeout_seconds"] > 0

    def test_the_shipped_reranker_is_the_identity_it_was_gated_into(self):
        """The nine-feature pass failed its held-out gate; that must stay recorded."""
        import json

        path = FROZEN_CONFIG_PATH.parent / "reranker.json"
        if not path.exists():
            pytest.skip("no fitted reranker artifact in the tree")
        payload = json.loads(path.read_text(encoding="utf-8"))
        provenance = payload.get("provenance", {})
        if provenance.get("shipped_tuned_weights"):
            return  # a future refit may clear the bar; then the weights are real
        assert payload["weights"][0] == 1.0
        assert all(w == 0.0 for w in payload["weights"][1:]), (
            "identity was expected: the bootstrap rejected the tuned weights"
        )
        assert provenance["dev_bootstrap"]["significant"] is False


class TestEncoderHonoursFrozenConfig:
    def test_accuracy_tier_selects_the_submission_model(self):
        encoder = CodeEncoder(tier="accuracy")
        assert encoder.model_name == ACCURACY_MODEL

    def test_interactive_tier_is_the_default_so_the_service_stays_light(self):
        encoder = CodeEncoder()
        assert encoder.tier == "interactive"
        assert encoder.model_name == INTERACTIVE_MODEL

    def test_unknown_tier_is_rejected(self):
        with pytest.raises(ValueError, match="tier must be"):
            CodeEncoder(tier="enormous")

    @pytest.mark.parametrize("tier", ["accuracy", "interactive"])
    def test_both_tiers_share_one_recipe(self, tier):
        """A tier may change which model runs; never how it is prompted."""
        encoder = CodeEncoder(tier=tier)
        assert encoder.query_prefix == APPS_QUERY_PREFIX
        assert encoder.document_prefix == ""
        assert encoder.max_seq_length == PUBLISHED_MAX_SEQ_LENGTH
        assert encoder.add_eos_token is True
        assert encoder.normalize_embeddings is True

    @pytest.mark.parametrize("tier", ["accuracy", "interactive"])
    def test_query_mode_prepends_the_instruction_and_document_mode_does_not(self, tier):
        encoder = CodeEncoder(tier=tier)
        assert encoder.get_prompt_text("abc", mode="query") == APPS_QUERY_PREFIX + "abc"
        assert encoder.get_prompt_text("abc", mode="document") == "abc"


@pytest.mark.skipif(
    pytest.importorskip("mteb", reason="mteb not installed") is None, reason="mteb not installed"
)
class TestAgainstMtebRegistry:
    """Pin the config against MTEB's own definition of the published model."""

    @staticmethod
    def _loader_kwargs(model_name: str) -> dict:
        import mteb

        meta = mteb.get_model_meta(model_name)
        return dict(getattr(meta, "loader_kwargs", {}) or {})

    def test_instruction_template_matches(self, frozen):
        kwargs = self._loader_kwargs(ACCURACY_MODEL)
        assert kwargs["instruction_template"] == F2LLM_INSTRUCTION_TEMPLATE

    def test_task_instruction_matches(self):
        from mteb.models.model_implementations import codefuse_models

        assert codefuse_models.f2llmv2_prompts_dict["AppsRetrieval"] == APPS_INSTRUCTION

    def test_passages_carry_no_instruction(self):
        assert self._loader_kwargs(ACCURACY_MODEL)["apply_instruction_to_passages"] is False

    def test_eos_and_sequence_cap_match(self, frozen):
        kwargs = self._loader_kwargs(ACCURACY_MODEL)
        assert kwargs["add_eos_token"] is True
        assert kwargs["max_seq_length"] == frozen["encoder"]["max_seq_length"]

    def test_pinned_revisions_match_the_registry(self, frozen):
        import mteb

        for section, name in (
            ("encoder", ACCURACY_MODEL),
            ("interactive_encoder", INTERACTIVE_MODEL),
        ):
            assert frozen[section]["revision"] == mteb.get_model_meta(name).revision, section


class TestBatchPlanner:
    """Attention memory is ``batch * seq**2``; budgeting it linearly OOMed twice."""

    def test_every_index_appears_exactly_once(self):
        lengths = [10, 4000, 30, 8192, 12, 500]
        batches = plan_batches(lengths, attention_budget=8192**2, max_batch=64)
        assert sorted(i for b in batches for i in b) == list(range(len(lengths)))

    def test_a_single_long_input_gets_its_own_batch(self):
        # Index 3 is the 8192-token outlier; batching it with the short inputs
        # is what produced the 32 GiB attention allocation.
        batches = plan_batches([8, 8, 8, 8192], attention_budget=8192**2, max_batch=64)
        assert next(b for b in batches if 3 in b) == [3]

    def test_quadratic_budget_is_respected(self):
        lengths = [100] * 200
        budget = 100 * 100 * 8
        for batch in plan_batches(lengths, attention_budget=budget, max_batch=64):
            cost = max(lengths[i] for i in batch) ** 2 * len(batch)
            assert cost <= budget

    def test_a_long_input_shrinks_the_batch_quadratically(self):
        """Doubling the sequence length must quarter the batch, not halve it."""
        short = plan_batches([512] * 256, attention_budget=512**2 * 16, max_batch=1024)
        long = plan_batches([1024] * 256, attention_budget=512**2 * 16, max_batch=1024)
        assert len(short[0]) == 16
        assert len(long[0]) == 4

    def test_an_input_over_budget_still_gets_encoded_alone(self):
        batches = plan_batches([8192], attention_budget=1000, max_batch=64)
        assert batches == [[0]]

    def test_max_batch_is_respected(self):
        for batch in plan_batches([1] * 500, attention_budget=10**12, max_batch=16):
            assert len(batch) <= 16

    def test_empty_input(self):
        assert plan_batches([]) == []


class TestRecipeVerification:
    """The loaded model is checked, not trusted."""

    @staticmethod
    def _encoder_with(tokenizer):
        encoder = CodeEncoder(tier="interactive")
        model = type("FakeModel", (), {})()
        model.tokenizer = tokenizer
        model.max_seq_length = encoder.max_seq_length
        return encoder, model

    def test_a_tokenizer_that_drops_the_eos_is_refused(self):
        class NoEosTokenizer:
            eos_token_id = 99
            add_eos_token = False

            def encode(self, text, add_special_tokens=True):
                return [1, 2, 3]

        encoder, model = self._encoder_with(NoEosTokenizer())
        with pytest.raises(RuntimeError, match="without an appended EOS"):
            encoder._verify_recipe_applied(model)

    def test_a_tokenizer_repaired_by_the_attribute_is_accepted(self):
        class RepairableTokenizer:
            eos_token_id = 99

            def __init__(self):
                self.add_eos_token = False

            def encode(self, text, add_special_tokens=True):
                return [1, 2, 99] if self.add_eos_token else [1, 2, 3]

        encoder, model = self._encoder_with(RepairableTokenizer())
        encoder._verify_recipe_applied(model)  # must not raise

    def test_a_silently_truncating_model_is_refused(self):
        class GoodTokenizer:
            eos_token_id = 99
            add_eos_token = True

            def encode(self, text, add_special_tokens=True):
                return [1, 2, 99]

        encoder, model = self._encoder_with(GoodTokenizer())
        model.max_seq_length = 512
        with pytest.raises(RuntimeError, match="max_seq_length is 512"):
            encoder._verify_recipe_applied(model)

    def test_a_correct_model_passes(self):
        class GoodTokenizer:
            eos_token_id = 99
            add_eos_token = True

            def encode(self, text, add_special_tokens=True):
                return [1, 2, 99]

        encoder, model = self._encoder_with(GoodTokenizer())
        encoder._verify_recipe_applied(model)


class TestExecutionFlagResolution:
    """`--no-execution` must actually be able to turn it off."""

    @staticmethod
    def _resolve(argv, defaults):
        import argparse

        parser = argparse.ArgumentParser()
        group = parser.add_mutually_exclusive_group()
        group.add_argument("--execution", dest="execution", action="store_true", default=None)
        group.add_argument("--no-execution", dest="execution", action="store_false")
        parser.add_argument("--execution-candidates", type=int, default=None)
        args = parser.parse_args(argv)
        enabled = defaults["execution_rerank"] if args.execution is None else args.execution
        return enabled, args.execution_candidates or defaults["execution_candidates"]

    def test_config_default_is_used_when_no_flag_is_given(self):
        from scripts.evaluate_mteb import get_pipeline_config

        defaults = get_pipeline_config()
        assert self._resolve([], defaults)[0] is defaults["execution_rerank"]

    def test_no_execution_overrides_an_enabled_config(self):
        defaults = {"execution_rerank": True, "execution_candidates": 10}
        assert self._resolve(["--no-execution"], defaults)[0] is False

    def test_execution_overrides_a_disabled_config(self):
        defaults = {"execution_rerank": False, "execution_candidates": 10}
        assert self._resolve(["--execution"], defaults)[0] is True

    def test_candidate_count_can_be_overridden(self):
        defaults = {"execution_rerank": True, "execution_candidates": 10}
        assert self._resolve(["--execution-candidates", "5"], defaults)[1] == 5
