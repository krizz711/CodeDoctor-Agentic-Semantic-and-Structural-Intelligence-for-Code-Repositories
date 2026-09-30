# C4-E1 addendum A1 — setup and exposure before decision runs

Written 2026-09-30. The original [protocol](PREREG.md) was committed as
`be0578f` before any C4 query encoding. The optional query adapter and
synthetic conversion runner were committed as `ed1c173`; train-only dataset
and stage-1 runner were committed as `4428522`. No C4 decision run or model
conversion has succeeded yet.

The pinned llama.cpp `b10269` tag resolves to full commit
`1c3c9674de4d455f1e571bed808252af54932767`. Its tagged server source
accepts numeric token arrays, returns `data[0].embedding` and evaluated-token
count at `/v1/embeddings`, and can reuse prompt prefixes. Our adapter supplies
HF token IDs after Sentence Transformers 5.0's whole-string `.strip()`,
starts one localhost slot with GPU offload 0 and RAM cache disabled, explicitly
erases that slot before every request, and includes both HTTP calls in timing.
Actual F32/Torch equivalence is still unverified. The pinned public tokenizer
alone returned terminal EOS 151645 on ordinary and empty-content probes.

The first private Kaggle synthetic smoke is recorded at
[smoke_v1.json](runs/smoke_v1.json) with its [raw log](runs/smoke_v1_infrastructure_failure.log).
It failed while pip resolved `cmake`: DNS resolution failed in the Kaggle
runtime, despite the notebook metadata reporting `enable_internet=true`.
No package install, model conversion, query encoding or quality measurement
completed. The permitted infrastructure retry has **not** run. Its prepared
notebook adds a DNS and HTTPS preflight plus explicit public PyPI index; verify
account Internet access before pushing it.

The owner later exposed the original APPS encoding notebook. We downloaded
only these files: corpus/query-train vectors, corpus/query-train IDs and texts,
train qrels, and the original encoding manifest. The original manifest
contains historical aggregate APPS test scores; they were seen during
provenance inspection, but no test query, vector or qrel was opened and no new
test result was computed. They are excluded from C4's new dataset. Its two
vector hashes match the handoff exactly:

- corpus `2d2b8938b80a4699887c3c839137d846c8d3e10e7bd533ad8bf71405efb2f99c`
- qtrain `9e19cdc2f32be6aff0b40c4d2305263e5a3a775383620122df8a2d1551dc2fa3`

The corpus texts from that notebook and C3-E2's dataset are identical parsed
rows despite different file hashes. The original train outcome stores define
disjoint 2,500-query TUNE and DEV assignments. `scripts/build_c4_data.py`
checked these against 5,000 train IDs, selected the registered seed's 100
TUNE queries, and created the private train-only dataset
[`lifemeansfun/codedoctor-c4-train-inputs`](https://www.kaggle.com/datasets/lifemeansfun/codedoctor-c4-train-inputs).
Its file hashes and assignment-store hashes are preserved in
[INPUTS_MANIFEST.json](INPUTS_MANIFEST.json), copied byte-for-byte from
`build/c4_data/MANIFEST.json`.
This preparation did not score or encode a C4 treatment.

Once the conversion smoke succeeds, the [stage-1 runner](../../scripts/c4_stage1.py)
and [Kaggle notebook builder](../../scripts/build_c4_stage1_kernel.py) are ready.
The stage-1 notebook is not to be launched on failed or missing conversion
controls. It uses the train-only dataset and the smoke output. Stage 2's live
replay and ranking analysis remains to be implemented after stage 1 passes;
the user requested a pause before that complex part to increase reasoning
depth. Do not weaken the registered gates if time or compute runs short.

## Infrastructure retry submitted

The user confirmed phone verification and Internet enabled in the editor.
Version 2 was submitted with the prepared DNS/HTTPS preflight and public PyPI
index. See [attempt record](runs/smoke_v2.json). It is the one permitted
infrastructure retry; a submitted/running state establishes no model result.

Before Stage 1, the scorer was hardened to reject zero-norm treatment/control
vectors explicitly: NaN cosine from a zero vector must never pass the F32
threshold checks. Six synthetic invalid-vector cases (zero/NaN/Inf for both
F32 and Q8) pass alongside the three existing gate tests. No decision data
was used for this correction.

## Closure

User stopped C4 on 2026-09-30 to finish the submission. Retry passed network
preflight and reached the smoke command, whose Torch reference worker exited
1 before comparison. No fidelity or speed result supports adoption. Preserve
this experiment as stopped/incomplete; no further C4 jobs are authorized.
