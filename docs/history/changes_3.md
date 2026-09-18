# CodeDoctor — GPU Acceleration Addendum (Colab T4)

**Scope:** this document only. Does not replace `CODEDOCTOR-V4-REMEDIATION-WORK-ORDER.md` — it extends it with GPU acceleration for the offline encoding path, plus the throughput diagnosis flagged after the ablation run.

**Correction to the request that produced this document:** nothing in this project is trained or fine-tuned. `THEME1-BUILD-PLAN-v3.pdf` explicitly excludes it: *"Explicitly not building: a new embedding model... We do not train or fine-tune anything."* `codefuse-ai/F2LLM-v2-0.6B` is used exactly as published. What is actually needed is **GPU-accelerated inference** — running the existing encoder faster on a T4 in `running.ipynb`, not training a model. If any step below starts building a training loop, optimizer, or loss function, stop — that is out of scope.

---

## 0. The constraint that shapes this whole document

The theme states the solution must run on CPU. The v1 plan settled this explicitly for the offline corpus-encoding step: *"The theme requires the solution to run on CPU; it does not require the offline corpus encoding to be slow on purpose."*

That draws a hard line:

| Path | Where it runs | GPU allowed? |
|---|---|---|
| Workstream A — encoding the 8,765-doc APPS test corpus, the 5,000-row train split, T6 ablation runs, A8 final release run | Your machine, offline, once | **Yes.** This is the one place GPU acceleration is fully sanctioned by the plan. |
| Workstream B — `B5` incremental re-index of the live JavaScript repo, anything the jury runs during the 15 October hands-on round | The jury's machine, live, no GPU present | **No.** Must work, and must be sub-second per B5's pass condition, on CPU alone. |

Consequence: **GPU code goes into the offline evaluation/ablation scripts only, behind an availability check, and must never become a hard dependency of `app/retrieval/encode.py`'s service path or the Docker image the jury runs.** If GPU code leaks into the service path, T11 (cold-start reproduction on a machine that's never run the project) and C1 (Docker reproducibility) both break the moment someone without a GPU follows the README.

---

## 1. Fix the actual bug before trusting any timing number

The earlier measurement — 550 seconds for one 64-document batch, ~9 seconds per document — is roughly 20–30x slower than the plan's own verified expectation of "well under an hour" for the full corpus on CPU. That gap does not go away by moving to GPU; it gets hidden by it. Fix it first, on CPU, so the number you eventually report is the encoder actually working correctly, not a slow bug running fast.

Check and fix, in `app/retrieval/encode.py` (or wherever `CodeEncoder` builds the model):

1. **`model.eval()` and `torch.no_grad()` (or `torch.inference_mode()`) around every encode call.** If gradient tracking is active during inference, that alone can roughly double compute and memory traffic for no reason.
2. **Tokenizer `max_length` matches real content, not a blanket 1024.** Average document is 573.5 characters (~100–150 tokens); average query 1,669.8 characters (~250–400 tokens). If every input is padded to 1024 regardless of actual length, that's needless multi-x overhead per batch. Set `max_length` to something like 512, with truncation, and confirm via a printed histogram of actual token counts that truncation only bites the documented tail (longest query 5,742 chars, longest doc 289,048 — outliers, not the norm).
3. **The model is constructed once, not per batch or per call.** Confirm `CodeEncoder.__init__` builds the model once and every `.encode()` call reuses the same in-memory instance.
4. **On CPU specifically:** check what `torch.get_num_threads()` actually returns in the running process versus how many cores the box has, and set `torch.set_num_threads(n)` explicitly if it's under-using available cores.

**Acceptance for this step:** re-run the same 64-document single-batch timing on CPU, unchanged hardware, after these four fixes. Report the new per-document time. If it hasn't dropped meaningfully, the bug isn't one of these four things — say so explicitly rather than moving on.

---

## 2. Add GPU acceleration to the offline path only

### Task 2.1 — Device-aware `CodeEncoder`, CPU still the default and the requirement

In `app/retrieval/encode.py`:

```python
import torch

def _select_device(preferred: str | None = None) -> str:
    if preferred:
        return preferred
    return "cuda" if torch.cuda.is_available() else "cpu"
```

- `CodeEncoder` takes an explicit `device` argument, defaulting to CPU unless the caller opts into GPU. **Do not make GPU the silent default anywhere the service path could pick it up.**
- The service construction path (whatever `app/server.py` or the live indexer uses) must construct `CodeEncoder` with `device="cpu"` explicitly, or omit the argument and rely on the CPU default — either way, add a test asserting the service-path encoder never ends up on `cuda` even when one is present, so a future edit can't accidentally regress this.
- The offline scripts (`scripts/evaluate_mteb.py`, `scripts/run_ablation.py`, anything doing the full-corpus encode) accept `--device cuda` / `--device cpu` / `--device auto`, defaulting to `auto` (use GPU if present, else CPU, print which was chosen).

### Task 2.2 — Batch size and precision tuned for a T4 (16GB)

- Raise batch size from whatever CPU used (likely 64) to something that uses the T4's memory properly — start at 256 or 512 and back off on OOM. Log the batch size actually used.
- Optionally enable mixed precision (`torch.autocast(device_type="cuda", dtype=torch.float16)`) for the encode forward pass. **If you do this, verify score parity in §3.2 below — fp16 can shift NDCG@10 by a small amount, and that shift needs to be measured, not assumed away.**
- Pin the CUDA/PyTorch versions actually available in the Colab T4 runtime into `requirements-eval.txt` or a separate `requirements-gpu.txt`, so the offline path is reproducible too, just not required for the service.

### Task 2.3 — `running.ipynb` structure

The notebook should, in order:
1. Confirm GPU: `torch.cuda.is_available()`, `torch.cuda.get_device_name(0)` — printed, not assumed.
2. Clone/pull the repo, install `requirements.txt` + `requirements-eval.txt` (+ `requirements-gpu.txt` if added).
3. Run the §1 fixed encoder against the **full** APPS test corpus (8,765 docs) and train split (5,000 rows) with `--device cuda`.
4. Save `appsretrieval_results.json`, the full ablation matrix output, and the recall-ceiling numbers as notebook outputs and also written back to files the repo can pick up (don't leave results stranded only in Colab's ephemeral session — download or push them back to the repo/artifact store before the runtime disconnects).

---

## 3. Test scenarios — GPU-specific

### 3.1 — Throughput, before/after, same document count

Run the fixed encoder (post-§1) on the same 64-document batch on CPU, then the same 64 documents on the T4:

```
CPU (fixed):  X sec/doc
GPU (T4):     Y sec/doc
```

**Expected:** both numbers should already be far better than the original 8.6 sec/doc once §1's bugs are fixed — if CPU alone gets into the tens-of-milliseconds-per-doc range, the full 8,765-doc corpus finishes in well under the "under an hour" the plan predicted, and GPU becomes a convenience, not a requirement. Report both numbers honestly rather than only the GPU one — this tells you whether §1 or the GPU is actually doing the work.

### 3.2 — Score parity, CPU vs GPU

Run the full T4 sanity-floor check (`codefuse-ai/F2LLM-v2-0.6B` against the real APPS test split) once on CPU and once on GPU:

```
CPU  ndcg_at_10 = ?
GPU  ndcg_at_10 = ?
```

**Expected:** within a very small tolerance (a few thousandths of NDCG@10) — larger drift means the fp16/autocast path from Task 2.2 is changing results meaningfully, in which case either drop autocast for the submission run or explicitly note the precision used in the deck. **The number that ships in `appsretrieval_results.json` should be labeled with which device produced it**, and ideally the final, submitted run is CPU (matching what the theme actually requires to be reproducible) even if GPU was used to iterate faster during the week.

### 3.3 — Full-corpus T6 ablation, no longer scoped down

With GPU making the full 8,765-doc run tractable in minutes rather than 21 hours, re-run the ablation matrix (base encoder / +prefixes / +fusion) on the **full** corpus rather than the 64-doc focused sample from the earlier report. A 64-document sample is too small to produce a trustworthy paired-bootstrap confidence interval; the full corpus is what the deck's ablation table should actually be built on.

**Expected:** same structure as before — every shipped component's CI excludes zero — but computed on real scale now that it's affordable.

### 3.4 — Service path never touches CUDA

```python
def test_service_encoder_defaults_to_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)  # pretend GPU exists
    encoder = get_shared_encoder()  # however the service constructs its encoder
    assert next(encoder.model.parameters()).device.type == "cpu"
```

**Expected:** passes even when a GPU is available in the test environment, proving the service path's CPU default isn't just "no GPU was found," it's an explicit choice that survives a GPU being present.

### 3.5 — B5's sub-second re-index, still CPU, still real

Re-run B5's single-chunk incremental re-index timing (from the earlier diagnosis) on CPU, after the §1 fixes, independent of anything GPU-related:

```
Single function edit → re-index wall clock = ? sec
```

**Expected:** under 1 second, per B5's actual pass condition. This is the number that matters for the hands-on round, and it must be demonstrated on CPU because that's what the jury will run. **If §1's fixes don't get this under a second on their own, the encoder is still too large or too slow for the live path regardless of what GPU acceleration does for the offline corpus, and that's a separate problem to raise, not one GPU access solves.**

---

## 4. What changes in the deliverables

- `appsretrieval_results.json` — regenerated on the full real corpus, device noted in the run log.
- `config/frozen.yaml` — add a `device: cpu` field (or equivalent) documenting that the frozen, submitted configuration is CPU-verified, even if GPU was used to iterate.
- Ablation table in the deck — built from the full-corpus run (§3.3), not the 64-document sample.
- README — two documented paths: "fast offline iteration (GPU, optional, `running.ipynb`)" and "reproducible submission run (CPU, required, matches what the jury's machine will do)." T11's cold-start test must still pass using only the CPU path with no GPU present.

---

## 5. Non-goals, again

- No training, no fine-tuning, no new checkpoint being produced. The model weights that ship are exactly `codefuse-ai/F2LLM-v2-0.6B` as published.
- No GPU dependency added to `app/server.py`, the Dockerfile, or anything the jury runs live.
- Do not skip §1. If the throughput bug is real, GPU will mask it in the offline path while leaving it fully present in the live CPU path that actually gets judged.