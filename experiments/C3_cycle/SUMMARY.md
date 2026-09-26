# Cycle 3: summary (quick reference; state on 2026-09-26, branch `v3`)

Everything was pre-registered before its decision data was touched (`PREREG.md`, addenda A1–A3). **DEV decides**; the APPS test split and the EXT-lockbox were never opened. Full numbers are in each experiment's `*_result*.json` / `.md`.

## What was run, and what it showed
| # | goal (brief / FAQ) | experiment | baseline | result | verdict |
|---|---|---|---|---|---|
| P0 | FAQ: "should ideally generalize" | Why execution added +0.000 on CodeContests (CC-onesol) | CC-onesol headroom 0.021 vs APPS 0.081 | APPS cut to the same 217-document pool: dense 0.902 → 0.989, execution gain +0.076 → +0.007 | **Small-pool ceiling effect, not a transfer failure.** The 3 residual misses are checker-dependent. |
| E1 | P0 screening; generalisation | Blank-line fallback: re-run a failed candidate on the sample without blank lines | DEV stdin NDCG@10 **0.95993** | **0.96771 (+0.0078** [+0.0031, +0.0131], p = 0.001, 11 up / 1 down); all families +0.0028; placebo and retry controls 0 | **Adopted** in `candidate_v2` |
| E1-B | generalisation (fidelity) | Judge-like I/O: stdin as a file, `exit`/`quit` bound | DEV 0.95327 | DEV 0 changes (83 re-run); EXT-dev +0.0069 n.s. (in-sample); CI green on all 3 OSes | **Accepted** (fidelity fix, no score claim) |
| E2 | FAQ A/B: CPU, efficiency, "how fast" | Int8 (OpenVINO) 1.7B query encoder, plus a single-machine end-to-end cost | DEV pipeline **0.95327**; 7.83 s / 8.5 GB per query | **Run 1 VOID** (S2: Kaggle gave a slower Intel Xeon). Disclosed, not a result: both int8 variants fail fidelity (median cosine 0.982 / 0.991 < 0.995) and run 1.3–1.4× *slower*; RSS 5.1 GB vs 11.5 GB. **Run 2 (the permitted repeat) finished on Kaggle, output not yet read** (HANDOFF A4 B1) | pending |
| E3 | P1: "rebuild … in a reasonable amount of time" | Incremental re-index in the demo, measured end to end | full rebuild at the same operating point | Parts 1–2 done (below). **Part 3 timing not run:** needs ≥ 3 GB free RAM, and moving it to a new machine needs addendum A4 first (HANDOFF A4 B2) | pending |

## Fixed on the way (correctness, found by this cycle's reviews)
- **The demo cache served another model's vectors** after a same-dimension model or prefix change. It is now keyed on the full recipe, with 3 regression tests. The scored-path keys are byte-identical to before.
- **The demo could not re-index an edited folder without a restart.** `rebuild=true` now re-reads it and re-encodes only the chunks that changed (tested: exactly the changed chunks; a pure move encodes 0).
- **The explorer's re-index silently turned dense search off after an edit.** It now takes `compute_embeddings` and says so in its response.

## Ideas tested and dropped (so nobody re-runs them)
| idea | why it was dropped |
|---|---|
| Small query model mapped into the 1.7B space (ridge, fused, vec2vec) | DEV 0.896 < plain 0.6B 0.901 (reproduced independently) |
| torch dynamic int8; int4 weights | vectors destroyed (cosine 0.21; 0.53–0.60) |
| 0.6B → 1.7B cascade | 56% escalate; saves 7–9% |
| Dense pseudo-relevance feedback | worse at every setting |
| Crash demotion; perturbation tie-break; identifier bonus | −0.012; −0.039 on the affected statements; null |
| LLM "spec cards" per snippet | ~0 gain for strong embedders in prior work; wrong target class |
| Bonus near-miss tie-break (MBR-exec) | promising pilot (0.46 vs chance 0.33) → **next cycle** |

## Mistakes I made and caught
- **A wrong claim about macOS pipes** (from the skeptic's man-page reading). The CI failure showed macOS reports buffered bytes on pipes. The test was corrected; arm B itself was unaffected.
- **My first P1 claim rule was arithmetically impossible,** and my first E2 efficiency rule was trivially passable. The plan reviewers caught both before any data was touched.
- **The E2 scoring script initially re-executed about 16k pairs the policy never uses,** and its fidelity sample was drawn only from timeouts. The known-answer control caught both; fixed and re-checked (Δ = 0, 0 fills).
