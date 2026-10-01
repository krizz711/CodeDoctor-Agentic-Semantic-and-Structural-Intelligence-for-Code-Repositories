# Submission handoff

## Current state — 2026-10-01 authorized publication

The user reports that the judges extended the deadline and approved amendments,
and explicitly authorized GitHub publication on2026-10-01. This supersedes the
earlier local-only restriction. The original submission was `408a61f`; see
[publication record](../docs/submission/PUBLICATION.md).
Work is on `fix/submission-corrections-local`. See the
[correction record](../docs/submission/LOCAL-CORRECTIONS.md) and committed
[protocol amendment](../docs/submission/PROTOCOL.md). Historical sections below
are retained as chronology, not current completion claims.

The four CPU query shards and private correction-P0 version 2 completed.
The accepted current CPU result is **NDCG@10 0.96583, MRR@10
0.9582509116971268, Recall@10 0.98805**. Authentic evidence is in
[results/final/p0](../results/final/p0/README.md); acceptance and both attempts
are preserved in `results/local-corrections/kaggle-attempt-{1,2}/`.

Version 1 failed at test collection before model/APPS loading. Version 2 was
the single infrastructure retry, measured clean commit `18c4b6c`, and passed
273 focused Linux tests (one optional-fixture skip), the real 1.7B cold Python
functional check, complete pinned-task coverage and 30 seeded live re-encodings
(minimum/median cosine 1.0). All 56 P0 and 39 cold-check recorded source hashes
match the local candidate. No fallback vectors or execution spawn failures.

The P0 runner used 2,609.959 seconds and peak process-tree RSS 9.41 GB; it is
saved-vector scoring replay with 30 live checks. CPU query encoding elapsed
11,100.779 / 10,147.842 / 12,613.804 / 13,987.302 seconds in separate earlier
jobs; do not claim these are full-live retrieval latency or a speedup.
NDCG and recall increased slightly over historical GPU P0, while MRR decreased
slightly. No significance claim, experiment attribution, test-score tuning,
new evaluation, Q8/OpenVINO adoption or EXT-lockbox access follows.

Local README, protocol/register, deck/PDF and delivery backups include this
result. Historical GPU P0 and older synthetic P1/Bonus keep their identities.
Human disclosure sign-off and video narration/missing demonstrations remain
outstanding; no signature or completed-video assertion is invented. Remote CI
is checked during publication; Docker verification remains separate. GitHub
publication and moving the required submission tag are now authorized.

The user ended C4. The original handoff remains at research-archive-2026-09-30.
Current reporting plan: [protocol](../docs/submission/PROTOCOL.md).
Research outcomes: [register](REGISTER.md); [archive](../archive/README.md).

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
