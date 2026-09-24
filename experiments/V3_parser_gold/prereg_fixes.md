# V3b — two parser correctness fixes (pre-registration, 2026-09-24)

Written before either fix was implemented or measured.

## What the V3-oracle round actually showed
- **The pre-registered verdict stands:** CodeContests CONF precision 0.910, below the 0.95 bar.
- **A post-hoc breakdown changes its meaning.** 117 of the 151 unconfirmed CONF extractions have exactly the public test's input and the statement's own output. That output is a valid answer that differs from DeepMind's stored one (multi-answer problems, float formatting).
  - Counting those as correct gives 0.980 CONF / 0.968 DIAG.
  - The oracle's "public tests = statement examples" assumption fails on multi-answer problems. That was not anticipated and is reported as post-hoc.
- **Real defect classes in APPS train** (the distribution FAQ 4 says the unseen set resembles):
  - **Brace arrays read as Python sets (F11):** 7 call-style statements (q1865, q4407, q4659, q206, q2927, q3095, q4905). All 7 are C-style arrays, not Python sets.
  - **Interactive statements whose sample is a judge transcript:** 2 (q1677, q1692, both with an "Interaction with the judge" section). Five other statements mention "interaction" in the story and are *not* interactive.
  - **HackerEarth input-format prose and CodeContests HTML artefacts:** 0 in APPS. **Not fixed:** it would be CodeContests-specific engineering with no APPS benefit (FAQ 4).

## Fixes
1. In call-style examples, a brace display `{a, b, c}` is read as a **list in source order**. Dict displays `{k: v}` are unchanged.
2. A statement with an explicit judge-interaction marker ("Interaction with the judge", "This is an interactive problem") yields **no stdin examples**.

## Ceiling (why this is not a decision experiment)
The fixes change at most 9 of 5,000 APPS train statements. Even a perfect effect is below the +0.003 bar. They are correctness and determinism fixes, and no score gain is claimed.

## Acceptance
- **Regression tests,** each failing before its fix:
  - the 7 set statements parse to lists, identically under two different `PYTHONHASHSEED` values;
  - q1677/q1692-style statements yield no stdin examples;
  - statements that merely mention "interaction" keep their examples;
  - dict displays stay dicts.
- **Oracle (the pre-registered metric), run once after the fixes:** CONF precision and recall not lower than before by more than 0.002. Both halves reported.
- **The full test suite passes.**
- Outcome stores are not re-recorded for 9 statements. The retrieval effect is bounded by 9/2,500 per half, and it is disclosed.

## Result (run once after both fixes; `result_oracle_after_fixes.json`)
| | before (V3) | after (V3b) | rule |
|---|---|---|---|
| CodeContests CONF precision | 0.9103 | 0.9134 | not lower ✓ |
| CodeContests CONF recall@cap | 0.9795 | 0.9776 | −0.0019, within −0.002 ✓ |
| APPS CONF stdin precision / recall | 0.9377 / 0.9583 | 0.9377 / 0.9583 | ✓ |
| APPS CONF call precision / coverage | 0.4342 / 0.2605 | 0.4348 / 0.2612 | ✓ |
| DIAG CodeContests precision / recall | 0.8856 / 0.9719 | 0.8903 / 0.9699 | reported |

- **Negative control:** still 0.0.
- **The recall dip is by design:** interactive statements no longer yield examples.
- **Regression tests:** `tests/test_parser_fixes.py`. All 7 behaviour tests failed before the fixes. A first version of the determinism test used integers, which hash identically in every process, so it passed pre-fix; it was rewritten with strings.
- **Accepted:** `execution_skip_interactive` is on in `config/candidate_v2.yaml` and off in the frozen config.

**Process note:** this pre-registration was written before either fix was
implemented or measured, but it was not committed before the oracle re-run,
which breaks the rule adopted this same round (STRESS_TEST F9). Its timing
therefore rests on the session record, not on git.
