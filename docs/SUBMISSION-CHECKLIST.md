# Submission checklist

This checklist follows `theme1_guidelines.pdf` and the verbatim jury FAQ in
`CLAUDE.md`. The current CPU submission has no validated current P0 result yet.
The root JSON and CSVs preserve a historical GPU-query run and must remain labeled
historical until a complete, source-matched final report is available.

## Deliverables and current state

| Deliverable | Current state | Evidence or action |
| --- | --- | --- |
| Runnable source and setup instructions | Present; current clean-environment validation must be attached | `README.md`, `scripts/README.md`, portable `requirements-submission.txt` |
| Current test-split MTEB JSON | Pending | Official `AppsRetrieval`, `cache=None`, frozen choice, one logged reporting run |
| Current ranked-response CSV | Pending | Generate from the same MTEB predictions; brief p.2 explicitly asks for responses CSV |
| Metrics CSV and execution audit | Historical root files present; current files pending | Attach matching run outputs and fallback counts |
| PPT and optional matching PDF | Local artifacts must be checked against current evidence | `VITV_Doctors_Submission.pptx`, `VITV_Doctors_Submission.pdf` |
| Demo video | Original local recording verified at 346.133 seconds; public link unchanged | Five-minute maximum is in supplied PPT slide 11; inspect the local cut preview and narration |
| AI usage form | Generated official form; human review and sign-off pending | `LangAI3.0_AI_Disclosure.docx` |
| GitHub Release and required run artifacts | Existing submitted release at `408a61f`; corrections are local only | A corrected release needs a validated run and explicit permission; do not move the submitted tag automatically |

The brief's explicit materials are JSON, PPT and runnable repository instructions;
it also requests response CSV and reviews a demo video during hands-on. P1 and
Bonus are evaluated there. A JSON format check is not score reproduction.

## Evidence required before the final decision

- Record exact source commit and hashes, frozen config hash, model revision,
  dataset revision/split, commands, timestamps, exit codes, device, hardware,
  package versions and raw logs.
- Separate CPU query encoding, scoring replay and example execution time.
  Reusing a CPU checkpoint is not a live-latency measurement or MTEB result cache.
  GPU query vectors cannot substantiate the CPU operating point.
- Report full NDCG@10 and MRR values with their operating point. Do not combine
  1.7B accuracy with 0.6B latency. The historical 0.96558/0.95857 result used GPU
  query encoding and the original frozen configuration.
- Report P1 medians, changed/reused chunks, fresh-rebuild equivalence and resources
  on the measured fixture. Current retained evidence uses source `823aa87` and
  a 2.20 GHz Xeon (4 logical/2 physical CPUs), three repetitions, 40 functions.
- Label Bonus as a synthetic 12-lineage/three-version correctness fixture:
  12/12 correct variants first versus 1/12 without verification. This is not
  unseen-dataset accuracy or a general 12-fold accuracy gain.
- Quote relevant FAQ answers, record objections and the cheapest falsifying
  checks, and preserve the explicit team boundary for test access/final decisions.
  The jury will rerun MTEB and use an internal unseen Python dataset.

Raw retained P1/Bonus evidence is under `results/hands-on/`; document preprocessing
and measured freezes are under `results/documents/`. These records predate the
current fixes and do not establish current benchmark parity.

## Disclosures that must stay aligned

The repository discloses plausible APPS/CodeContests training exposure through
OpenCodeReasoning-2 and xCodeEval, with exact per-problem exposure unverified.
Earlier execution-depth and numeric-comparison choices used the test split.
The FAQ accepts benchmark-exposed models, prefers documentation, and warns against
overfitting; static parsing and dynamic execution are allowed.

The FAQ permits GPU pre/post processing, including incremental updates, and asks
for minimal GPU retrieval (ideally CPU). It counts retrieval APIs as GPU usage.
It permits non-incremental P1 approaches while evaluating time and resources.
Permissible licenses are preferred; non-commercial models are not disqualified.
Upstream licenses still apply, and this repository has no top-level license grant.

Opening an arbitrary folder creates a single `working` snapshot; it does not
import external Git history. Demonstrate version comparison with a built-in
history target and label the model tier. Passing worked examples does not prove
correctness. The subprocess runner limits resources without network/filesystem
isolation. The AI form must not claim a human attestation or digital signature
until the representative supplies it.

## Release preparation

The PDF requires a GitHub Release and artifacts. The official submission portal
screenshot supplied by the user additionally requires the exact tag
`PRISM_GENAI_HACKATHON_Y2026`; it says the tagged commit is judged and every
referenced submission material must be present there. The absence of that name
from the theme PDF does not override the supplied portal requirement.
This local checklist does not create a tag, upload artifacts or make anything public.

Attach the final run's MTEB JSON, response CSV, metrics CSV, execution audit and
reproduction evidence plus files needed to run. Include the current PPT, demo
and reviewed AI form as required by the submission portal. Verify every required
local path, use the measured source commit, and keep README/deck/video/metrics
consistent. Archive superseded results with their historical operating point.
