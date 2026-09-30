"""V3b parser fixes (experiments/V3_parser_gold/prereg_fixes.md).

1. Statements write arrays C-style, ``{10, 3, -27, -1}``.  Read as a Python
   set they lost order and duplicates, and iterated in hash order, so the same
   statement produced different examples in different processes.
2. An interactive statement's sample is a judge transcript, not stdin.
"""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

from app.pipeline.execution_rerank import ExecutionConfig, _statement_examples
from app.pipeline.statement import parse_call_examples, parse_literal

REPO_ROOT = Path(__file__).resolve().parent.parent

KATA = """Given an array of integers, return the minimum sum.

```
minSum({10, 3, -27, -1}, 3) => -30
minSum({5, 5, 1}, 1) => 1
```
"""

WORDS = """Count the words.

```
countWords({"lie!", "cannot", "butts", "I", "big", "like", "and"}) => 7
countWords({"a", "b"}) => 2
```
"""

INTERACTIVE = """Guess the number.

-----Interaction with the judge-----
Read the judge's reply after each query.

-----Example-----
Input
3
1

Output
? 2
! 3
"""

STORY = """Devu wants inter-gender interaction in his class to be maximum.

-----Example-----
Input
2 3

Output
5
"""


class TestBraceArrays:
    def test_braces_are_lists_in_source_order(self):
        assert parse_literal("{10, 3, -27, -1}") == (True, [10, 3, -27, -1])
        assert parse_literal("{'b', 'a', 'b'}") == (True, ["b", "a", "b"])

    def test_nested_and_dicts(self):
        assert parse_literal("[{1, 2}, {3}]") == (True, [[1, 2], [3]])
        assert parse_literal("{'a': {1, 2}}") == (True, {"a": [1, 2]})
        assert parse_literal("{}") == (True, {})

    def test_call_examples_use_lists(self):
        calls = parse_call_examples(KATA)
        assert calls and calls[0].args == ([10, 3, -27, -1], 3)

    def test_duplicates_survive(self):
        calls = parse_call_examples(KATA)
        assert calls[1].args == ([5, 5, 1], 1)  # a set used to collapse this to {1, 5}

    def test_the_same_statement_parses_the_same_in_every_process(self):
        # Strings, because only str/bytes hashes are randomised per process:
        # a set of small ints iterates in the same order everywhere.
        code = ("from app.pipeline.statement import parse_call_examples as p; import sys; "
                "sys.stdout.write(repr([(c.args, c.expected) for c in p(sys.stdin.read())]))")
        outputs = set()
        for seed in ("1", "2", "3", "4", "5"):
            env = {**os.environ, "PYTHONHASHSEED": seed}
            done = subprocess.run([sys.executable, "-c", code], input=WORDS, capture_output=True, text=True,
                                  cwd=REPO_ROOT, env=env, check=True)
            outputs.add(done.stdout)
        assert len(outputs) == 1 and "lie!" in outputs.pop()


class TestInteractive:
    CONFIG = ExecutionConfig(enabled=True, call_style=True)

    def test_off_by_default_keeps_the_submitted_behaviour(self):
        stdin, _ = _statement_examples(INTERACTIVE, self.CONFIG)
        assert stdin  # frozen/default path unchanged

    def test_interactive_statements_have_no_stdin_examples(self):
        stdin, calls = _statement_examples(INTERACTIVE, replace(self.CONFIG, skip_interactive=True))
        assert stdin == [] and calls is None

    def test_a_story_that_mentions_interaction_keeps_its_examples(self):
        stdin, _ = _statement_examples(STORY, replace(self.CONFIG, skip_interactive=True))
        assert [(e.stdin.strip(), e.expected_stdout.strip()) for e in stdin] == [("2 3", "5")]
