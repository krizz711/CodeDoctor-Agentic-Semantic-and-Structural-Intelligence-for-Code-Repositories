"""Development splits must be random, reproducible, and distribution-matched."""

from __future__ import annotations

import numpy as np
import pytest

from app.pipeline.query_analysis import QueryFamily
from app.pipeline.splits import (
    TEST_DOMINANT_FAMILY,
    family_distribution,
    make_development_split,
)

COMPETITIVE = (
    "Find the answer.\n\n-----Input:-----\nThe first line contains $n$.\n\n"
    "-----Output:-----\nPrint the answer.\n\n-----Sample Input:-----\n2\n\n"
    "-----Sample Output:-----\n4\n"
)
KATA = "Your task is to write a function which doubles a number. ```double(n)```"


@pytest.fixture
def mixed_queries() -> list[str]:
    """Ordered by source, exactly like the real corpus."""
    return [COMPETITIVE] * 200 + [KATA] * 200


class TestDistributionMatching:
    def test_family_filter_keeps_only_the_target_family(self, mixed_queries):
        split = make_development_split(mixed_queries, match_family=TEST_DOMINANT_FAMILY)
        assert len(split.eligible) == 200
        assert (split.eligible < 200).all()

    def test_no_family_filter_keeps_everything(self, mixed_queries):
        split = make_development_split(mixed_queries, match_family=None)
        assert len(split.eligible) == len(mixed_queries)

    def test_target_family_is_the_competitive_one(self):
        assert TEST_DOMINANT_FAMILY is QueryFamily.COMPETITIVE_STDIN

    def test_too_few_eligible_queries_raises_rather_than_returning_junk(self):
        with pytest.raises(ValueError, match="too few"):
            make_development_split([KATA] * 50, match_family=TEST_DOMINANT_FAMILY)


class TestSplitMechanics:
    def test_halves_are_disjoint_and_cover_the_eligible_set(self, mixed_queries):
        split = make_development_split(mixed_queries, match_family=None)
        assert set(split.tune).isdisjoint(split.dev)
        assert np.array_equal(
            np.sort(np.concatenate([split.tune, split.dev])), split.eligible
        )

    def test_same_seed_reproduces_the_split(self, mixed_queries):
        a = make_development_split(mixed_queries, match_family=None, seed=7)
        b = make_development_split(mixed_queries, match_family=None, seed=7)
        assert np.array_equal(a.tune, b.tune)
        assert np.array_equal(a.dev, b.dev)

    def test_different_seeds_give_different_splits(self, mixed_queries):
        a = make_development_split(mixed_queries, match_family=None, seed=7)
        b = make_development_split(mixed_queries, match_family=None, seed=8)
        assert not np.array_equal(a.tune, b.tune)

    def test_split_is_not_contiguous(self, mixed_queries):
        """A contiguous cut of a source-ordered corpus is not exchangeable."""
        split = make_development_split(mixed_queries, match_family=None, seed=3)
        midpoint = len(mixed_queries) // 2
        assert (split.tune >= midpoint).any(), "TUNE must draw from both halves"
        assert (split.dev < midpoint).any(), "DEV must draw from both halves"

    def test_dev_fraction_is_honoured(self, mixed_queries):
        split = make_development_split(mixed_queries, match_family=None, dev_fraction=0.25)
        assert abs(len(split.dev) / len(split.eligible) - 0.25) < 0.02


class TestFamilyDistribution:
    def test_fractions_sum_to_one(self, mixed_queries):
        distribution = family_distribution(mixed_queries)
        assert sum(distribution.values()) == pytest.approx(1.0)

    def test_reports_the_actual_mix(self, mixed_queries):
        distribution = family_distribution(mixed_queries)
        assert distribution[QueryFamily.COMPETITIVE_STDIN.value] == pytest.approx(0.5)
        assert distribution[QueryFamily.KATA_FUNCTION.value] == pytest.approx(0.5)

    def test_empty_input_does_not_divide_by_zero(self):
        assert family_distribution([]) == {}
