"""Decide which retrieval passes a query needs, instead of running them all.

The pipeline has four passes and they do not cost the same thing:

    exact       ~0.1 ms   inverted index lookup, no model
    bm25        ~2 ms     postings list, no model
    dense       seconds   one forward pass through a 0.6B encoder on CPU
    execution   ~2 s x N  runs candidate programs in a sandbox

Running all four on every query makes the cheap case cost as much as the
expensive one. It is also why the system looks slow pointed at a repository:
a question naming ``authenticate_user`` is answered exactly by an index
lookup, and then waits on an embedding that cannot change the answer.

This module plans instead. It reads cheap signals off the query, runs the free
passes first, looks at the evidence they produced, and escalates only when that
evidence is weak. Every decision carries a reason string, because a planner
whose choices you cannot inspect is indistinguishable from a hardcoded
if-statement.

The gate that matters most is execution verification. It needs worked examples
in the query text, and measurement says that is a property of the DATASET, not
of the wording: 3676/3765 AppsRetrieval queries carry them (97.6%) and 0/180
CodeTransOceanDL queries do (0.0%). ``parse_worked_examples`` answers that in
microseconds, so the pass is cheap to skip and expensive to run blind.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

from app.pipeline.execution_rerank import parse_worked_examples
from app.retrieval.hybrid import normalize_identifier

#: Rough wall-clock cost class per pass. Reporting only -- the router gates on
#: evidence, not on a cost model.
PASS_COST = {
    "exact": "free",
    "bm25": "free",
    "structural": "cheap",
    "dense": "expensive",
    "execution": "expensive",
}

#: "who calls X", "what imports Y" -- questions about the graph, not the text.
_STRUCTURAL_INTENT = re.compile(
    r"\b(call(s|ed|ers?|ing)?|invoke[sd]?|import(s|ed|ing)?"
    r"|referenc(e|es|ed)|used\s+by|depend(s|ency|encies)?\s+on"
    r"|who\s+(calls|uses|imports))\b",
    re.IGNORECASE,
)

#: dotted.path, snake_case, camelCase, CONSTANT, or an explicit call foo()
_SYMBOL_LIKE = re.compile(
    r"[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]*"
    r"|[a-z0-9]+_[a-z0-9_]+"
    r"|[a-z]+[A-Z][A-Za-z0-9]*"
    r"|\b[A-Z][A-Z0-9_]{2,}\b"
    r"|[A-Za-z_][A-Za-z0-9_]*\s*\(\s*\)"
)

#: Below this many words a query is a lookup, not a description.
_PROSE_MIN_WORDS = 4


@dataclass(frozen=True)
class Step:
    """One pass the router chose to run, or chose not to."""

    name: str
    reason: str

    @property
    def cost(self) -> str:
        return PASS_COST.get(self.name, "unknown")


@dataclass
class RouterSignals:
    """What the router observed about the query, before running anything."""

    word_count: int = 0
    symbols: tuple[str, ...] = ()
    has_worked_examples: bool = False
    structural_intent: bool = False
    dense_available: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "word_count": self.word_count,
            "symbols": list(self.symbols),
            "has_worked_examples": self.has_worked_examples,
            "structural_intent": self.structural_intent,
            "dense_available": self.dense_available,
        }


@dataclass
class Decision:
    """An escalation decision taken after looking at real evidence."""

    escalated: bool
    reason: str
    top_margin: Optional[float] = None
    exact_hit: bool = False


@dataclass
class RetrievalPlan:
    """The passes to run, the ones deliberately skipped, and why."""

    steps: list[Step] = field(default_factory=list)
    skipped: list[Step] = field(default_factory=list)
    signals: RouterSignals = field(default_factory=RouterSignals)
    decisions: list[Decision] = field(default_factory=list)

    @property
    def step_names(self) -> list[str]:
        return [step.name for step in self.steps]

    def ran(self, name: str) -> bool:
        return any(step.name == name for step in self.steps)

    def add(self, name: str, reason: str) -> None:
        if not self.ran(name):
            self.steps.append(Step(name, reason))
        # A pass that was deferred and then escalated into is no longer
        # "skipped". Leaving it in both lists is how a plan lies about itself.
        self.skipped = [step for step in self.skipped if step.name != name]

    def skip(self, name: str, reason: str) -> None:
        if self.ran(name):
            return
        self.skipped.append(Step(name, reason))

    def as_dict(self) -> dict[str, Any]:
        return {
            "ran": [
                {"pass": s.name, "cost": s.cost, "reason": s.reason} for s in self.steps
            ],
            "skipped": [
                {"pass": s.name, "cost": s.cost, "reason": s.reason} for s in self.skipped
            ],
            "signals": self.signals.as_dict(),
            "decisions": [
                {
                    "escalated": d.escalated,
                    "reason": d.reason,
                    "top_margin": d.top_margin,
                    "exact_hit": d.exact_hit,
                }
                for d in self.decisions
            ],
        }


class QueryRouter:
    """Plans a retrieval, then decides whether the cheap result was enough.

    ``escalate_margin`` is the fused-score gap between rank 1 and rank 2 below
    which the lexical answer is treated as ambiguous and the dense pass earns
    its seconds. RRF scores are small -- each contributing method adds
    ``1 / (60 + rank)`` -- so the default is correspondingly small.
    """

    def __init__(
        self,
        *,
        escalate_margin: float = 0.004,
        always_dense: bool = False,
    ) -> None:
        self.escalate_margin = escalate_margin
        self.always_dense = always_dense

    # ---------------------------------------------------------------- signals
    def read_signals(self, query: str, *, dense_available: bool = False) -> RouterSignals:
        symbols = tuple(
            dict.fromkeys(match.group(0).strip() for match in _SYMBOL_LIKE.finditer(query))
        )
        return RouterSignals(
            word_count=len(query.split()),
            symbols=symbols,
            has_worked_examples=bool(parse_worked_examples(query)),
            structural_intent=bool(_STRUCTURAL_INTENT.search(query)),
            dense_available=dense_available,
        )

    # ------------------------------------------------------------------- plan
    def plan(self, query: str, *, dense_available: bool = False) -> RetrievalPlan:
        """The passes to run before any evidence exists."""
        signals = self.read_signals(query, dense_available=dense_available)
        plan = RetrievalPlan(signals=signals)

        # The free passes always run. They cost less than deciding not to.
        plan.add("exact", "inverted-index lookup, no model")
        plan.add("bm25", "postings list, no model")

        match = _STRUCTURAL_INTENT.search(query)
        if match:
            plan.add(
                "structural",
                "query asks about the call graph, not the text "
                "(matched {0!r})".format(match.group(0)),
            )
        else:
            plan.skip("structural", "no caller/import/reference intent in the query")

        if not signals.dense_available:
            plan.skip("dense", "no warm embeddings for this snapshot; lexical answers now")
        elif self.always_dense:
            plan.add("dense", "always_dense=True (routing disabled)")
        elif signals.word_count >= _PROSE_MIN_WORDS and not signals.symbols:
            plan.add(
                "dense",
                "{0}-word description with no identifier to match on".format(
                    signals.word_count
                ),
            )
        elif signals.symbols:
            # Deferred, not refused: escalate() may still call for it.
            plan.skip(
                "dense",
                "query names {0} identifier(s) ({1}); deferred pending lexical evidence".format(
                    len(signals.symbols), ", ".join(signals.symbols[:3])
                ),
            )
        else:
            plan.skip(
                "dense",
                "{0}-word lookup; deferred pending lexical evidence".format(
                    signals.word_count
                ),
            )

        if signals.has_worked_examples:
            plan.add("execution", "query carries worked examples; candidates can be verified")
        else:
            plan.skip(
                "execution",
                "no parseable worked examples in the query; nothing to verify against",
            )

        return plan

    # --------------------------------------------------------------- escalate
    def escalate(
        self,
        plan: RetrievalPlan,
        results: Sequence[Any],
    ) -> Decision:
        """Given the cheap results, decide whether dense is worth running.

        This is what makes it a planner rather than a lookup table: the
        decision reads evidence that did not exist when ``plan`` was built.
        """
        if plan.ran("dense"):
            return self._record(plan, Decision(False, "dense already in the plan"))

        if not plan.signals.dense_available:
            return self._record(plan, Decision(False, "dense unavailable; cannot escalate"))

        if not results:
            decision = Decision(True, "lexical passes returned nothing")
            plan.add("dense", "escalated: no lexical candidates")
            return self._record(plan, decision)

        exact_hit = any("exact" in getattr(result, "sources", ()) for result in results[:3])
        margin: Optional[float] = None
        if len(results) >= 2:
            margin = float(results[0].score) - float(results[1].score)

        # The strongest stop signal is not the margin -- RRF gaps are tiny and
        # shrink further when several near-duplicates match -- it is whether
        # rank 1 IS the symbol the query named. A dense pass cannot improve on
        # having already found the function by name.
        named = {normalize_identifier(symbol) for symbol in plan.signals.symbols}
        top_name = normalize_identifier(getattr(results[0].document, "name", "") or "")
        if named and top_name and top_name in named:
            return self._record(
                plan,
                Decision(
                    False,
                    "rank 1 is {0!r}, the identifier the query named; "
                    "dense cannot improve on an exact name match".format(
                        results[0].document.name
                    ),
                    top_margin=margin,
                    exact_hit=exact_hit,
                ),
            )

        if exact_hit and margin is not None and margin >= self.escalate_margin:
            return self._record(
                plan,
                Decision(
                    False,
                    "exact symbol match at the head and a clear margin "
                    "({0:.5f} >= {1}); dense cannot change rank 1".format(
                        margin, self.escalate_margin
                    ),
                    top_margin=margin,
                    exact_hit=True,
                ),
            )

        if margin is not None:
            why = "top-2 margin {0:.5f} < {1}".format(margin, self.escalate_margin)
        else:
            why = "only one lexical candidate"
        if not exact_hit:
            why += "; no exact symbol match at the head"

        plan.add("dense", "escalated: " + why)
        return self._record(
            plan, Decision(True, why, top_margin=margin, exact_hit=exact_hit)
        )

    @staticmethod
    def _record(plan: RetrievalPlan, decision: Decision) -> Decision:
        plan.decisions.append(decision)
        return decision
