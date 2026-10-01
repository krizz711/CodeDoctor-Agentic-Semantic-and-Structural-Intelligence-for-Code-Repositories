# Submission handoff

## Current state — 2026-10-01 local corrections

The submitted release is at `408a61f`; the user explicitly forbids GitHub changes.
Their 2026-10-01 clarification permits the private Kaggle CPU evaluation.
Work is on `fix/submission-corrections-local`. See the
[correction record](../docs/submission/LOCAL-CORRECTIONS.md) and committed
[protocol amendment](../docs/submission/PROTOCOL.md). Historical sections below
are retained as chronology, not current completion claims.

All four CPU query shard jobs completed and their raw outputs were downloaded.
No validated current CPU P0 score exists. Strict artifact/provenance/result
gates, demo tier matching, dependency setup, documentation, PPT/PDF and official
AI form corrections are local. Final1.7B checks require more than this Mac's
8 GiB; the prepared private Kaggle continuation is now authorized to launch.
Human disclosure sign-off, video narration/missing demo
review and Samsung's amendment permission remain external decisions. Never move
the official tag or upload a local correction automatically.

The user ended C4 and authorized a finished submission on main.
The original handoff is preserved at tag research-archive-2026-09-30.
Current reporting plan: [protocol](../docs/submission/PROTOCOL.md).
Research outcomes: [register](REGISTER.md); [archive](../archive/README.md).
No new final score exists until the final MTEB run completes.

## Finalization progress, 2026-09-30

Runtime preparation: `d3ce432`, branch `finalize/submission`. Local checks and
GitHub CI passed as detailed in `results/validation/README.md`.
Kaggle account: lifemeansfun. Renewed CLI authorization after first pushes failed
before notebook creation. Documents v1 and hands-on v1 are running:
- https://www.kaggle.com/code/lifemeansfun/codedoctor-submission-documents
- https://www.kaggle.com/code/lifemeansfun/codedoctor-submission-hands-on

Next: download/verify complete document cache and hands-on artifacts, run four CPU
query shards, verify coverage and matching source, then final P0 scoring once.
Staged query checkpointing is explicitly authorized in the reporting protocol.
README and template-based deck now emphasize the pre/post-processing architecture.
The deck currently labels final metrics pending; do not publish it as measured.
Team-member details and demo video remain intentionally assigned to the team.

### Kaggle Hands-On Validation & Query Shards Update (2026-09-30)

- Fixed indexing range in `scripts/validate_hands_on.py` line 135 (`values[4:14]`).
- Re-built and launched `codedoctor-submission-hands-on` (version 2): **COMPLETED**.
  - P1 incremental indexing: 100% equivalence, ~68x speedup on unchanged code.
  - Bonus cross-version retrieval: 100% correct variant selection with execution verification vs 8.33% control baseline.
- Launched 4 disjoint CPU query encoding shards on Kaggle (`queries-0` to `queries-3`): RUNNING.
- Added compliance materials: `requirement.txt`, `VITV_Doctors_Submission.pptx`, `VITV_Doctors_Submission.pdf`, `LangAI3.0_AI_Disclosure.docx`.
- Finalized README and slide deck for judge evaluation.
