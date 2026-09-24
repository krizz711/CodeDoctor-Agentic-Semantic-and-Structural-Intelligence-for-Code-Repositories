# V3-oracle — the statement parser against labels its author did not write

Written 2026-09-24, before any oracle data was downloaded or matched.

## Why this replaces the human sheet (for now)
V3 asked for a teammate who did not write the parser to label 207 statements.
None is available. Claude wrote the parser, so Claude's labels would share its
blind spots. Instead the ground truth comes from people who built public
datasets for other purposes:

- **O1, APPS train** (in distribution): `codeparrot/apps` `train.jsonl`, field
  `input_output` = the problem's official tests (stdin pairs, or `fn_name` +
  argument lists). Matched to CoIR train queries by whitespace-normalised
  statement text. The APPS test file is never downloaded.
- **O2, CodeContests train** (out of distribution): `deepmind/code_contests`,
  field `public_tests` = exactly the sample tests shown in each statement,
  recorded separately by DeepMind. Only the columns `name`, `description`,
  `public_tests` and `source` are read. Problems whose statements
  near-duplicate an APPS statement (the MinHash dedup in
  `scripts/build_external_set.py`) are removed. The CodeContests valid/test
  splits (EXT-dev, EXT-lockbox) are not used, so the lockbox stays sealed.

## Frozen parser
Commit `6cbffed`; blob hashes `statement.py 33852241`,
`execution_rerank.py e2760526`, `query_analysis.py d933d434`. The result JSON
records the hashes it ran with.

**Addendum (written after reading the config code, before any matching ran).**
- **Primary configuration:** the one the v2 candidate runs — `call_style` on,
  `extended_stdin` off, `max_examples = 2` (the `ExecutionConfig` default; the
  scored path does not override it). Recall uses `min(visible tests, 2)` per
  statement as its denominator, because the parser is told to stop at two.
- **Secondary:** `extended_stdin` on, `max_examples = 10`, to show what the
  parser finds without the cap.
- Call-style values are compared with `values_equal` at tolerance 1e-6, so
  float formatting does not count as an error.
- Diagnostic only (added after the matcher's unit tests, before any oracle
  data was scored): stdin precision with all whitespace collapsed, because an
  input `2   3` and an official `2 3` are the same input to any program. The
  pre-registered rule above stays the decision metric.

## Split
Each oracle set is split 50/50 by a seeded hash of the problem id (seed
20260924) into **DIAG** and **CONF**. The frozen parser is scored on both and
both numbers are reported. Any parser change may be motivated only by DIAG
failures and is judged **once** on CONF. A second look at CONF is disclosed.

## Metrics
Stdin pairs are compared after the verifier's own `_normalize_output`. A test
is **visible** in a statement when its normalised input lines and output lines
each occur as a contiguous run of normalised statement lines, input before
output.

- **O2 (true labels).** Pair precision = extracted pairs equal to a public
  test / extracted pairs. Pair recall = visible public tests extracted /
  visible public tests. Also reported: statement-level exact-set rate,
  visibility rate, and the rate of "nothing extracted" on statements with
  visible tests.
- **O1 (tests ⊇ examples, usually).** Confirmed precision = extracted pairs
  equal to an official test / extracted pairs. This is a lower bound: an
  example absent from the official tests is not necessarily wrong. Visible
  recall is as for O2. For call-style problems: confirmed precision of
  (arguments, expected) against (`inputs`, `outputs`) with `values_equal`,
  accepting an output wrapped in a one-element list. Also reported: the
  coverage of statements with at least one confirmed example.
- Intervals: bootstrap over statements, 2,000 resamples, 95% percentile.
- Audit: 30 unconfirmed extractions and 30 missed visible tests per set, drawn
  by seed (not chosen by hand), printed verbatim in the result file.

## Decision rules
- **Pass** if pair precision is ≥ 0.95 on O2-CONF and confirmed precision is
  ≥ 0.95 on O1-CONF (point estimates; intervals reported). This is the plan's
  kill criterion for PRE-1.
- Recall below 0.90 on either CONF set is flagged as "action needed". It is not
  a kill criterion: a missed example costs verification coverage, while a
  wrong example can reject a correct program or promote a wrong one.
- Low precision on a family (by source or layout) with n ≥ 30 is reported per
  family even if the overall rate passes.

## Limits, stated in advance
- APPS confirmed precision understates true precision: statement examples are
  not always among the official tests.
- The visibility rule misses examples rendered inline (e.g. `Input: 3 4`). They
  are left out of the recall denominator, so recall is measured on
  line-rendered examples only.
- Claude still writes the matching code. It gets its own unit tests with
  hand-built cases, and the audit samples are printed so anyone can check it.

## Benchmarks (written before any oracle data was scored)
- **Negative control:** each statement is scored against another problem's
  tests (a seeded derangement within each oracle set). Stdin precision must be
  ≤ 0.05, and so must call precision where there are enough call examples;
  higher means the matcher confirms things it should not, and the whole result
  is void.
- **Visibility sanity:** at least 50% of O2 statements must have ≥ 1 visible
  public test. Otherwise the visibility rule is broken and recall is not
  reported.
- **Positive control:** the matcher's unit tests (`tests/test_parser_oracle.py`)
  pass before the run.
- **Runtime:** expected under 15 minutes; stopped and investigated after 30.
