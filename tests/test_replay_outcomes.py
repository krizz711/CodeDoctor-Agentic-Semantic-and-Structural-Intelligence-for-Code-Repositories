"""The replay scorer: comparators and the legacy ranking rule."""

from __future__ import annotations

from scripts.replay_outcomes import Policy, call_verdict, metrics, rerank, stdin_verdict

RULES_NONE = {"any_answer": False, "tolerance": None, "case_insensitive": False, "modulus": None}


def _run(out: str, rc: int = 0, status: str = "completed") -> dict:
    return {"status": status, "rc": rc, "exc": None, "out": out, "tracked": True, "read": 0, "eof": False}


class TestStdinVerdict:
    def test_exact_and_failure_classes(self):
        legacy = Policy("legacy")
        assert stdin_verdict(_run("5"), "5\n", RULES_NONE, legacy) == "strong"
        assert stdin_verdict(_run("5", rc=1), "5", RULES_NONE, legacy) == "none"
        assert stdin_verdict(_run("5", status="timeout"), "5", RULES_NONE, legacy) == "none"
        assert stdin_verdict(_run("6"), "5", RULES_NONE, legacy) == "none"

    def test_numeric_needs_the_policy(self):
        assert stdin_verdict(_run("1.0000001"), "1.0", RULES_NONE, Policy("legacy")) == "none"
        assert stdin_verdict(_run("1.0000001"), "1.0", RULES_NONE, Policy("n", numeric_tolerance=1e-6)) == "strong"

    def test_statement_tolerance(self):
        rules = dict(RULES_NONE, tolerance=1e-3)
        policy = Policy("s", statement_tolerance=True)
        assert stdin_verdict(_run("0.3334"), "0.3333", rules, policy) == "strong"
        assert stdin_verdict(_run("0.3334"), "0.3333", RULES_NONE, policy) == "none"

    def test_case_rule_only_when_stated(self):
        policy = Policy("c", case_rule=True)
        assert stdin_verdict(_run("yes"), "YES", dict(RULES_NONE, case_insensitive=True), policy) == "strong"
        assert stdin_verdict(_run("yes"), "YES", RULES_NONE, policy) == "none"

    def test_any_answer_rungs(self):
        rules = dict(RULES_NONE, any_answer=True)
        multiset = Policy("m", any_answer_multiset=True)
        shape = Policy("s", any_answer_shape=True)
        assert stdin_verdict(_run("3\n2 1 3"), "3\n1 2 3", rules, multiset) == "strong"
        assert stdin_verdict(_run("3\n2 1 4"), "3\n1 2 3", rules, multiset) == "none"
        assert stdin_verdict(_run("3\n2 1 4"), "3\n1 2 3", rules, shape) == "weak"
        assert stdin_verdict(_run("3\n2 1"), "3\n1 2 3", rules, shape) == "none"
        # without the statement's permission no relaxation applies
        assert stdin_verdict(_run("3\n2 1 3"), "3\n1 2 3", RULES_NONE, multiset) == "none"


def _entry(outs: list[str], legacy_examples: int = 1) -> dict:
    pool = [f"d{i}" for i in range(len(outs))] + ["d_far"]
    return {
        "qid": "q1",
        "family": "competitive_stdin",
        "rules": RULES_NONE,
        "legacy_examples": legacy_examples,
        "stdin_examples": [{"stdin": "1\n", "expected": "5"}],
        "call_examples": [],
        "distinct_expected": 0,
        "pool": pool,
        "scores": [1.0 - 0.1 * i for i in range(len(pool))],
        "candidates": [
            {"j": j, "doc": pool[j], "function_style": False, "stdin_runs": [_run(out)]}
            for j, out in enumerate(outs)
        ],
    }


class TestRerank:
    def test_verified_candidates_rise_in_dense_order(self):
        entry = _entry(["0", "5", "0", "5"])
        assert rerank(entry, Policy("legacy"))[:4] == ["d1", "d3", "d0", "d2"]

    def test_nothing_verified_keeps_dense_order(self):
        entry = _entry(["0", "1", "2"])
        assert rerank(entry, Policy("legacy")) == entry["pool"]

    def test_execution_depth_is_respected(self):
        entry = _entry(["0", "0", "5"])
        assert rerank(entry, Policy("legacy", exec_depth=2)) == entry["pool"]

    def test_extended_examples_need_the_flag(self):
        entry = _entry(["0", "5"], legacy_examples=0)
        assert rerank(entry, Policy("legacy")) == entry["pool"]
        assert rerank(entry, Policy("x", extended_stdin=True))[0] == "d1"

    def test_metrics(self):
        assert metrics(["a", "b"], "a") == (1.0, 1.0, 1.0, 1.0)
        ndcg, mrr, r1, r10 = metrics(["a", "b"], "b")
        assert round(ndcg, 6) == round(1 / 1.584962500721156, 6) and mrr == 0.5 and r1 == 0.0
        assert metrics(["a"], "z") == (0.0, 0.0, 0.0, 0.0)


class TestCallVerdict:
    def _entry(self, expected: list, results: list, distinct: int) -> tuple[dict, dict]:
        entry = {
            "rules": RULES_NONE,
            "call_examples": [
                {"args": repr((i,)), "kwargs": "()", "expected": repr(e), "callee": None, "layout": "x"}
                for i, e in enumerate(expected)
            ],
            "distinct_expected": distinct,
        }
        record = {"call_run": {"status": "completed", "exc": None, "results": results}}
        return entry, record

    def test_all_match_is_strong(self):
        entry, record = self._entry([10, 20], [("ok", "10", None, "f"), ("ok", "20", None, "f")], 2)
        assert call_verdict(record, entry, Policy("c", call_style=True)) == "strong"

    def test_one_mismatch_is_none(self):
        entry, record = self._entry([10, 20], [("ok", "10", None, "f"), ("ok", "21", None, "f")], 2)
        assert call_verdict(record, entry, Policy("c", call_style=True)) == "none"

    def test_low_entropy_guard(self):
        entry, record = self._entry([True, True], [("ok", "True", None, "f"), ("ok", "True", None, "f")], 1)
        assert call_verdict(record, entry, Policy("c", call_style=True)) == "none"

    def test_identity_guard(self):
        entry, record = self._entry([0, 1], [("ok", "0", None, "f"), ("ok", "1", None, "f")], 2)
        assert call_verdict(record, entry, Policy("c", call_style=True)) == "none"


class TestLiveComparatorMirrorsReplay:
    """The live comparator (judge_rules) must agree with the replay's verdicts,
    or a policy measured by replay would not be the policy that ships."""

    CASES = [
        ("5", "5", {}),
        ("1.0000001", "1.0", {}),
        ("0.3334", "0.3333", {"tolerance": 1e-3}),
        ("yes", "YES", {"case_insensitive": True}),
        ("yes", "YES", {}),
        ("3\n2 1 3", "3\n1 2 3", {"any_answer": True}),
        ("3\n2 1 4", "3\n1 2 3", {"any_answer": True}),
        ("3\n2 1 3", "3\n1 2 3", {}),
    ]

    def test_verdicts_agree(self):
        from app.pipeline.execution_rerank import ExecutionConfig, compare_output
        from app.pipeline.statement import JudgeRules

        policy = Policy("judge", numeric_tolerance=1e-6, statement_tolerance=True, case_rule=True,
                        any_answer_multiset=True)
        for produced, expected, rules in self.CASES:
            replay = stdin_verdict(_run(produced), expected, dict(RULES_NONE, **rules), policy)
            config = ExecutionConfig(enabled=True, numeric_tolerance=1e-6, judge_rules=True,
                                     statement_rules=JudgeRules(**rules))
            live = compare_output(produced, expected, config)
            assert (replay == "strong") == (live is not None), (produced, expected, rules, replay, live)


class TestDeepening:
    def _deep_entry(self, first: list[str], deep: list[str]) -> dict:
        entry = _entry(first + deep)
        for record in entry["candidates"]:
            if record["j"] >= len(first):
                record["deep"] = True
        return entry

    def test_deep_candidates_only_count_when_the_window_found_nothing(self):
        entry = self._deep_entry(["0", "5"], ["5"])
        policy = Policy("d", exec_depth=2, deepen_to=3)
        assert rerank(entry, policy)[0] == "d1"  # verified inside the window: deep one ignored

    def test_hard_and_soft_promotion(self):
        entry = self._deep_entry(["0", "1"], ["5"])
        assert rerank(entry, Policy("hard", exec_depth=2, deepen_to=3))[0] == "d2"
        soft = rerank(entry, Policy("soft", exec_depth=2, deepen_to=3, deep_soft=True))
        assert soft[:2] == ["d0", "d2"]

    def test_policies_without_deepening_ignore_deep_records(self):
        entry = self._deep_entry(["0", "1"], ["5"])
        assert rerank(entry, Policy("legacy", exec_depth=2)) == entry["pool"]
