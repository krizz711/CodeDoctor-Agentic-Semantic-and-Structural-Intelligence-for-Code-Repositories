"""Second-pass reranking: features, weights, and the fitter."""

from __future__ import annotations

import numpy as np
import pytest

from app.pipeline.code_analysis import CodeFamily, analyze_code
from app.pipeline.features import (
    FEATURE_NAMES,
    CorpusStatistics,
    FamilyCompatibility,
    FeatureExtractor,
    InverseDocumentFrequency,
)
from app.pipeline.metrics import metrics_from_ranks, paired_bootstrap
from app.pipeline.query_analysis import QueryFamily, analyze_query
from app.pipeline.reranker import Reranker, RerankerConfig
from app.pipeline.search import (
    build_candidate_pool,
    document_key,
    hubness_from_pool,
    load_hub_profile,
    save_hub_profile,
)
from app.pipeline.tuning import fit_weights, ndcg_at_10, ranks_for_weights


DOCS = [
    "n = int(input())\nprint(n * 2)",
    "class Solution:\n    def twoSum(self, nums):\n        return nums[0]",
    "def count_squares(lines):\n    return len(lines)",
    "import sys\nfor line in sys.stdin:\n    print(line)",
    "def helper(a, b):\n    return a + b",
]


class TestIdf:
    def test_rare_token_outweighs_common_one(self):
        idf = InverseDocumentFrequency(
            [frozenset({"common", "rare"})] + [frozenset({"common"})] * 99
        )
        assert idf.weight("rare") > idf.weight("common")

    def test_unseen_token_gets_the_maximum_weight(self):
        idf = InverseDocumentFrequency([frozenset({"a"})] * 10)
        assert idf.weight("never_seen") >= idf.weight("a")

    def test_overlap_is_zero_without_intersection(self):
        idf = InverseDocumentFrequency([frozenset({"a", "b"})])
        assert idf.overlap(frozenset({"a"}), frozenset({"z"})) == 0.0

    def test_overlap_is_one_when_query_side_is_covered(self):
        idf = InverseDocumentFrequency([frozenset({"a", "b"})] * 5)
        assert idf.overlap(frozenset({"a"}), frozenset({"a", "b"})) == pytest.approx(1.0)


class TestFamilyCompatibility:
    def test_unknown_query_family_contributes_nothing(self):
        compat = FamilyCompatibility()
        facts = analyze_query("do the thing")
        assert facts.family is QueryFamily.UNKNOWN
        assert compat.score(facts, analyze_code(DOCS[0])) == 0.0

    def test_estimation_prefers_the_observed_pairing(self):
        compat = FamilyCompatibility.estimate(
            [QueryFamily.COMPETITIVE_STDIN] * 50 + [QueryFamily.KATA_FUNCTION] * 50,
            [CodeFamily.STDIN_SCRIPT] * 50 + [CodeFamily.FUNCTION_LIBRARY] * 50,
        )
        matched = compat.table[(QueryFamily.COMPETITIVE_STDIN.value, CodeFamily.STDIN_SCRIPT.value)]
        mismatched = compat.table[
            (QueryFamily.COMPETITIVE_STDIN.value, CodeFamily.FUNCTION_LIBRARY.value)
        ]
        assert matched > mismatched

    def test_json_round_trip(self):
        original = FamilyCompatibility.estimate(
            [QueryFamily.KATA_FUNCTION] * 10, [CodeFamily.FUNCTION_LIBRARY] * 10
        )
        restored = FamilyCompatibility.from_json(original.to_json())
        assert restored.table == original.table


class TestFeatureExtractor:
    def test_feature_matrix_shape_and_dense_passthrough(self):
        stats = CorpusStatistics.build(DOCS)
        extractor = FeatureExtractor(stats)
        facts = analyze_query("write a function ```count_squares(lines)``` that counts squares")
        candidates = np.array([2, 0, 1], dtype=np.int32)
        scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)
        features = extractor.for_query(facts, candidates, scores)
        assert features.shape == (3, len(FEATURE_NAMES))
        assert np.allclose(features[:, FEATURE_NAMES.index("dense")], scores)

    def test_defined_name_hit_fires_on_the_named_function(self):
        stats = CorpusStatistics.build(DOCS)
        extractor = FeatureExtractor(stats)
        facts = analyze_query("write a function ```count_squares(lines)``` please")
        features = extractor.for_query(
            facts, np.array([2, 0], dtype=np.int32), np.array([0.5, 0.5], dtype=np.float32)
        )
        column = FEATURE_NAMES.index("defined_name_hit")
        assert features[0, column] == 1.0  # doc 2 defines count_squares
        assert features[1, column] == 0.0

    def test_empty_candidate_list_returns_empty_matrix(self):
        extractor = FeatureExtractor(CorpusStatistics.build(DOCS))
        features = extractor.for_query(
            analyze_query("anything"), np.array([], dtype=np.int32), np.array([], dtype=np.float32)
        )
        assert features.shape == (0, len(FEATURE_NAMES))


class TestSearch:
    def test_pool_is_sorted_and_finds_the_exact_match(self):
        rng = np.random.default_rng(0)
        docs = rng.normal(size=(50, 16)).astype(np.float32)
        docs /= np.linalg.norm(docs, axis=1, keepdims=True)
        queries = docs[[3, 17, 40]]
        pool = build_candidate_pool(queries, docs, depth=5)
        assert pool.indices[:, 0].tolist() == [3, 17, 40]
        for row in range(3):
            assert (np.diff(pool.scores[row]) <= 1e-6).all()

    def test_unnormalized_embeddings_are_rejected(self):
        docs = np.full((4, 8), 2.0, dtype=np.float32)
        with pytest.raises(ValueError, match="not L2-normalized"):
            build_candidate_pool(docs, docs, depth=2)

    def test_hubness_flags_a_document_that_wins_every_query(self):
        indices = np.zeros((100, 10), dtype=np.int32)
        indices[:, 1:] = np.arange(1, 10)
        from app.pipeline.search import CandidatePool

        pool = CandidatePool(indices=indices, scores=np.zeros((100, 10), dtype=np.float32))
        hub = hubness_from_pool(pool, n_documents=50, top_k=10)
        assert hub[0] == pytest.approx(1.0)
        assert hub[25] == 0.0

    def test_hub_profile_survives_a_corpus_reordering(self, tmp_path):
        """Index-aligned profiles break silently when the corpus is rebuilt."""
        texts = [f"def f{i}(): return {i}" for i in range(6)]
        hubness = np.array([0.0, 0.9, 0.0, 0.3, 0.0, 0.0], dtype=np.float32)
        path = save_hub_profile(tmp_path / "hub.npz", texts, hubness)

        shuffled = [texts[i] for i in (3, 0, 5, 1, 4, 2)]
        aligned = load_hub_profile(path, shuffled)
        assert aligned.tolist() == pytest.approx([0.3, 0.0, 0.0, 0.9, 0.0, 0.0])

    def test_unknown_document_scores_zero_not_a_penalty(self, tmp_path):
        path = save_hub_profile(
            tmp_path / "hub.npz", ["known"], np.array([1.0], dtype=np.float32)
        )
        assert load_hub_profile(path, ["known", "never seen"]).tolist() == [1.0, 0.0]

    def test_saving_a_mismatched_profile_is_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="hub scores"):
            save_hub_profile(tmp_path / "hub.npz", ["a", "b"], np.zeros(3, dtype=np.float32))

    def test_document_key_is_content_addressed(self):
        assert document_key("same") == document_key("same")
        assert document_key("same") != document_key("different")


class TestRerankerConfig:
    def test_default_is_identity(self):
        assert RerankerConfig().is_identity

    def test_wrong_weight_count_is_rejected(self):
        with pytest.raises(ValueError, match="expected"):
            RerankerConfig(weights=(1.0, 2.0))

    def test_json_round_trip_preserves_weights(self, tmp_path):
        weights = tuple(float(i) / 10 for i in range(len(FEATURE_NAMES)))
        config = RerankerConfig(weights=weights, candidate_depth=37)
        path = config.save(tmp_path / "reranker.json")
        restored = RerankerConfig.load(path)
        assert restored.weights == weights
        assert restored.candidate_depth == 37

    def test_missing_file_falls_back_to_identity(self, tmp_path):
        assert RerankerConfig.load(tmp_path / "absent.json").is_identity

    def test_stale_feature_set_is_rejected_loudly(self, tmp_path):
        path = tmp_path / "stale.json"
        path.write_text('{"feature_names": ["dense"], "weights": [1.0]}', encoding="utf-8")
        with pytest.raises(ValueError, match="different feature set"):
            RerankerConfig.load(path)


class TestReranker:
    def test_identity_config_is_an_exact_passthrough(self):
        stats = CorpusStatistics.build(DOCS)
        reranker = Reranker(RerankerConfig(), FeatureExtractor(stats))
        scores = np.array([[0.9, 0.1, 0.5, 0.2, 0.3]], dtype=np.float32)
        assert np.array_equal(reranker.rescore(scores, [analyze_query("x")]), scores)

    def test_reranked_block_never_falls_below_the_untouched_tail(self):
        stats = CorpusStatistics.build(DOCS)
        weights = [0.0] * len(FEATURE_NAMES)
        weights[FEATURE_NAMES.index("dense")] = 1.0
        weights[FEATURE_NAMES.index("defined_name_hit")] = 5.0
        config = RerankerConfig(weights=tuple(weights), candidate_depth=2)
        reranker = Reranker(config, FeatureExtractor(stats))
        scores = np.array([[0.9, 0.8, 0.1, 0.05, 0.02]], dtype=np.float32)
        out = reranker.rescore(scores, [analyze_query("write a function ```count_squares(lines)```")])
        # the two pool members stay above every document the pool did not reach
        assert min(out[0, 0], out[0, 1]) > max(out[0, 2], out[0, 3], out[0, 4])

    def test_mismatched_fact_count_is_rejected(self):
        stats = CorpusStatistics.build(DOCS)
        weights = [0.0] * len(FEATURE_NAMES)
        weights[0] = 1.0
        weights[3] = 0.5
        reranker = Reranker(RerankerConfig(weights=tuple(weights)), FeatureExtractor(stats))
        with pytest.raises(ValueError, match="query facts"):
            reranker.rescore(np.zeros((2, 5), dtype=np.float32), [analyze_query("x")])


class TestFitter:
    @staticmethod
    def _planted_signal(n_queries=300, depth=40, seed=0):
        rng = np.random.default_rng(seed)
        features = np.zeros((n_queries, depth, len(FEATURE_NAMES)), dtype=np.float32)
        features[:, :, 0] = np.sort(
            rng.random((n_queries, depth)).astype(np.float32), axis=1
        )[:, ::-1]
        gold = rng.integers(0, depth, size=n_queries).astype(np.int32)
        gold[:10] = -1
        for row in range(n_queries):
            if gold[row] >= 0:
                features[row, gold[row], FEATURE_NAMES.index("defined_name_hit")] = 1.0
        return features, gold, depth

    def test_dense_only_ranks_match_pool_positions(self):
        features, gold, depth = self._planted_signal()
        weights = np.zeros(len(FEATURE_NAMES), dtype=np.float32)
        weights[0] = 1.0
        ranks = ranks_for_weights(features, gold, weights, pool_miss_rank=depth + 1)
        expected = np.where(gold >= 0, gold + 1, depth + 1)
        assert np.array_equal(ranks, expected)

    def test_fitter_recovers_a_perfect_signal_up_to_the_pool_ceiling(self):
        features, gold, depth = self._planted_signal()
        fit = fit_weights(features, gold, pool_miss_rank=depth + 1, n_sweeps=3, verbose=False)
        ranks = ranks_for_weights(
            features, gold, np.asarray(fit.weights, dtype=np.float32), pool_miss_rank=depth + 1
        )
        ceiling = float((gold >= 0).mean())
        assert ndcg_at_10(ranks) == pytest.approx(ceiling, abs=1e-6)

    def test_fitter_never_recovers_a_query_the_pool_missed(self):
        features, gold, depth = self._planted_signal()
        fit = fit_weights(features, gold, pool_miss_rank=depth + 1, n_sweeps=3, verbose=False)
        ranks = ranks_for_weights(
            features, gold, np.asarray(fit.weights, dtype=np.float32), pool_miss_rank=depth + 1
        )
        assert (ranks[gold < 0] == depth + 1).all()

    def test_dense_weight_stays_pinned(self):
        features, gold, depth = self._planted_signal()
        fit = fit_weights(features, gold, pool_miss_rank=depth + 1, n_sweeps=2, verbose=False)
        assert fit.weights[FEATURE_NAMES.index("dense")] == 1.0


class TestMetrics:
    def test_ndcg_matches_the_closed_form_for_a_single_gold(self):
        ranks = np.array([1, 2, 3, 11], dtype=np.int32)
        metrics = metrics_from_ranks(ranks)
        expected = (1.0 + 1 / np.log2(3) + 1 / np.log2(4) + 0.0) / 4
        assert metrics.ndcg_at_10 == pytest.approx(expected)
        assert metrics.mrr_at_10 == pytest.approx((1 + 0.5 + 1 / 3) / 4)

    def test_rank_beyond_ten_scores_zero(self):
        assert metrics_from_ranks(np.array([11, 50, 101])).ndcg_at_10 == 0.0

    def test_identical_systems_are_not_significant(self):
        values = np.random.default_rng(0).random(500)
        assert not paired_bootstrap(values, values).significant

    def test_a_uniform_improvement_is_significant(self):
        baseline = np.random.default_rng(1).random(500)
        assert paired_bootstrap(baseline, baseline + 0.05).significant

    def test_mismatched_lengths_are_rejected(self):
        with pytest.raises(ValueError, match="align"):
            paired_bootstrap(np.zeros(5), np.zeros(6))
