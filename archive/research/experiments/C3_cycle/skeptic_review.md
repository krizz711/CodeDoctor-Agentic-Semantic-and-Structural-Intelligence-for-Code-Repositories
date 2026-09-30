# Skeptic review of the shortlisted experiments (2026-09-26)

Three independent agents each tried to refute one experiment's premises, using TUNE, code and web sources only. Full JSON is in `skeptic_reports.json`; scripts are in `skeptic_scripts/`. Every verdict was **go with changes**.

## E1: verifier fidelity
- **Confirmed:**
  - 24/777 TUNE stdin statements have interleaved blank lines. They come from the raw statement text (CodeChef rendering), not from our parser.
  - The APPS official inputs equal the **collapsed** sample in 24/24 cases.
  - Live: the gold verifies after the collapse in 21/24.
- **Refuted or qualified:**
  - One statement is harmed (q1090, 1.0 → 0.5): a short-number output coincidentally matches.
  - With 6 up / 1 down, the sign test gives only p = 0.125.
- **Design changes adopted:**
  - the any-blank detector (28 vs 21 gold recoveries on TUNE);
  - a consumption gate (non-gold verifications 13 → 9, 9 up / 0 down; chosen on TUNE, so in-sample until DEV);
  - a derangement placebo as the negative control;
  - arm B demoted to a fidelity fix: it changes 0 of 85 TUNE re-runs, and its EXT-dev 7/9 was chosen post hoc.
- **Arm B facts:**
  - The `os.fstat(0).st_size` idiom fails on pipes on Windows and Linux.
  - Binding `exit` reproduces CPython's return codes and output exactly.
  - Binding must happen in the runner's builtins; prepending source would break `from __future__`.

## E2: quantised query encoder
- **Refuted premise.** The research's 0.997 weight-only cosine does not describe either runtime. OpenVINO and llama.cpp Q8_0 both also quantise activations. On a 160M proxy, each step roughly doubles 1 − cos.
- **Refuted bar.** A cosine of 0.999 is too strict. Random perturbations down to cos 0.99 are NDCG-neutral on TUNE, while a dropped EOS gives 0.37. Cosine is therefore a validity gate, and DEV NDCG decides.
- **Unverified:**
  - The speedup. Prefill is compute-bound (about 136 GFLOP/s on 2 cores), so any gain needs int8 compute (VNNI/AMX) on the machine; the CPU instruction set has to be detected in-kernel.
  - The RAM saving (~1.7 GB of int8 weights vs 6.9 GB) is near-certain (estimate).
- **Traps found:**
  - F2LLM's EOS is `<|im_end|>` (151645); `<|endoftext|>` is the pad token.
  - An OpenVINO export without nncf silently stays in float32.
  - The document cache key has no backend field, so the int8 model must never encode documents.

## E3: P1
- **Confirmed:** the demo cache key has no model or recipe in it. Same-dimension changes serve stale vectors silently; 1.7B vs 0.6B crashes loudly.
- **New and more important:**
  - The workbench cannot re-index a changed folder in-process: `start_build` returns early.
  - The explorer's re-index skips embeddings, so dense retrieval switches off after any edit.
  - A timing experiment on the current code would therefore time a path the demo does not run.
- **Measured:**
  - 0.6B CPU: 1.275 s per document at batch 1; windows of 8 were about 4× slower (one run).
  - Mixing bf16 GPU seed vectors with CPU fp32 updates: top-10 order identity 0.9716, ΔNDCG@10 +3.9e-6.
