# Authorized submission amendment —2026-10-01

The user reports that the judges extended the submission period and approved
corrections, then explicitly instructed: “so yeah commit on github”. This is
the authorization to publish the already prepared corrections on main and update
`PRISM_GENAI_HACKATHON_Y2026` and its release assets. It supersedes the prior
local-only restriction. No exact new deadline is asserted. Repository visibility,
YouTube content, team attestations and signatures are not changed by this action.

## Scope and traceability

Original main/tag: `408a61f0f6e0c2a58528e88155f3dc6a4fa25a33`.
Corrected evidence/materials commit: `ec949bc6c0e9849d820f4c569155ca781754613a`.
Measured clean CPU source: `18c4b6cbd097437bfb8498fefe572cddb658bc42`.
Publication adds documentation of authorization and a CI check of the accepted
CPU JSON/source. Every56 P0 and39 cold-check source/config/dependency hash must
still match; no ranking/model changes or new APPS evaluation are authorized or
needed by this publication.

Accepted result: NDCG@10 **0.96583**, MRR@10 **0.9582509116971268**,
Recall@10 **0.98805**. Full evidence lives in `results/final/p0/`; the root
`appsretrieval_results.json` remains the historical GPU result. The release asset
named `appsretrieval_results.json` is the current CPU result from `results/final/p0/`.
Both ranked-response CSV and metrics CSV are updated accordingly. Historical
P1/Bonus remain synthetic fixture measurements from their original source;
no new generalization, latency or speedup claim is introduced.

The required tag moves transparently to the corrected main commit. Original
result/artifact identities remain recoverable at the original commit, and
previous release assets and metadata are archived locally before replacement.
Release notes identify the amendment and original tag target. No main-branch
history is force-pushed. The exact release commit and upload checksums are
verified after publication and retained in the local publication receipt.

## Requirements and objections

The theme brief requires response CSV, MTEB JSON in a GitHub Release, the supplied
presentation and reproducible setup. The portal requires the exact tag on the
judged commit. The complete relevant FAQ answers remain verbatim in `PROTOCOL.md`;
they support CPU retrieval, permissible offline GPU preprocessing, disclosure
of possible contamination and qualified fixture results. The FAQ does not specify
post-submission amendment policy; the user's report of judge approval supplies
that decision, rather than an invented FAQ permission.

The strongest case against publication is that the video still exceeds five
minutes and lacks complete dense/edit demonstrations, the AI form requires
human attribution/attestations/signature, and Docker is unverified. Publication
of verified code/evidence does not cure these issues or certify a complete
submission. The user remains responsible for those human tasks. The existing
public video link and unsigned form are preserved, with accurate limitations.

A valid slightly lower MRR is reported alongside the slightly higher NDCG and
recall; no significance or causal superiority claim follows. Prior APPS exposure
is retained separately from this hosted run. Cheapest falsifying checks before
publishing are strict official JSON validation, all measured blob hashes,
compressed prediction identity, deck/PDF checksums and CI. Full evaluation is
not repeated. A failed source/CI check must be resolved before moving the tag;
release downloads must exactly match the prepared local artifacts.
