# E2 handoff reconciliation — 2026-09-29

Status: **second available output is void under the original controls; E2 stops**.
Branch: `v3`, base `7d64a80`. Existing unrelated workspace changes preserved.

On 2026-09-29 the authenticated Kaggle CLI downloaded the notebook's current
`e2/e2_A.json`. The stored [copy](E2_runs/e2_A_run2_downloaded.json) has SHA256
`94183162d9786918c13a59ad4ee3569b2561ec5e1e4db632a6225dec8b2e9e3f`.
It differs from run 1 (`0c38a4f8…`) in raw timings. The handoff identifies
notebook version 2 as the finished repeat, and Kaggle lists the latest run on
2026-09-26 14:38:24. The CLI's output endpoint does not return a version number;
`kernels pull .../2` returned HTTP 403, so the exact version association is an
inference from the handoff and latest-run metadata, not API-certified provenance.
Neither APPS test nor EXT-lockbox was accessed.

The adjudicator command was:

```sh
.venv/bin/python scripts/reconcile_e2.py \
  build/kernel_e2a/out2/e2/e2_A.json --attempt 2 \
  --out build/e2_reconciliation/run2.json
```

It returned `status=void`, with both T1 and T1b failing S2 as reported and
recomputed, no missing controls. REF mean was 11.290 s for T1 and 11.456 s for
T1b, outside the fixed 7.832 s ±25% sanity band. Both also failed fidelity
(median cosine 0.982222 and 0.990780 against 0.995) and were slower than REF
(ratios 1.368 and 1.2695), but those are observations from a **void run**, not
a valid negative treatment result. CPU: Intel Xeon 2.20 GHz, AVX2/FMA, no
VNNI/AVX-512/AMX, 2 physical cores. Peak process RSS: 5.140 GB (T1) and
7.515 GB (T1b). The original A3 stop rule permits no third attempt. No
OpenVINO arm advances to stage B or candidate adoption.

## Completed preparation

- Installed Kaggle CLI 2.2.4 in the project virtual environment.
- Implemented `scripts/reconcile_e2.py`: checks the original stage-A controls,
  recomputes the reference timing sanity check, checks exported int8 constants,
  derives fidelity/efficiency gates, checks the fixed arm-selection order, and
  distinguishes void, incomplete, negative and gate-passed results. It never
  grants adoption; a gate pass still requires E2 stage B.
- Checked the stored run-1 artifact: **void**, failing S2 for both attempted
  arms, with no missing evidence. This is reanalysis, not a repeated run.
- The original unauthenticated download attempt failed. After the user
  configured credentials locally, `kernels status` succeeded and the current
  JSON and notebook log downloaded. Credentials were not copied into code,
  commands or records.

## Retrieval and provenance commands

```sh
.venv/bin/kaggle kernels output munalsingh/codedoctor-e2-ov-a \
  -p build/kernel_e2a/out2 --file-pattern '.*e2_A.json$'
```

**Provenance check:** this installed CLI's output endpoint requests the current
kernel output by owner and slug, not an explicit version number. Verify that
this is version 2 in Kaggle's version history and retain that metadata. If the
latest version differs, obtain version 2 explicitly through the notebook UI;
do not label an arbitrary latest output as run 2. The adjudicator cannot
certify remote provenance and rejects a byte-identical copy of known run 1.

The result copy and adjudication are preserved. Continue with C4-E1 Q8_0 under
its own preregistration. If conversion fails correctness checks, pause for the
user's requested reasoning-depth change.

## Checks actually run

- `pytest -q tests/test_reconcile_e2.py`: 9 passed.
- `ruff check scripts/reconcile_e2.py tests/test_reconcile_e2.py --select F401,F811,F821,E9`: passed.
- Stored run-1 assessment: `build/e2_reconciliation/run1.json`.

No model conversion, real-model smoke, Linux run, quality evaluation, timing
experiment, submission change or push has run during this preparation.
