# Submission handoff

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
