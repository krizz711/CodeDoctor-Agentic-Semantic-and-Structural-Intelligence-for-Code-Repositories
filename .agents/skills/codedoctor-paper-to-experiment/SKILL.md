---
name: codedoctor-paper-to-experiment
description: Map a research paper or literature question to CodeDoctor's actual retrieval paths and a falsifiable experiment proposal. Use for paper reproduction and research-to-code planning, not general repository onboarding.
---

# Paper to CodeDoctor experiment

Read `AGENTS.md` and `CLAUDE.md`, then inspect the code path the idea would
actually change. The scored MTEB pipeline and interactive demo are different.
Use `docs/CODEX-ANALYSIS.md` for routing, and check newer experiment records for
prior rejection or adoption before proposing another run.

Read the primary paper and official implementation when available. Verify
title/version, claimed contribution, datasets, baseline, metric, compute and
evaluation protocol. Label inaccessible sources and missing details; do not
fill them with invented citations, results or implementation choices.

Produce a compact mapping: paper claim -> relevant method/code -> CodeDoctor
integration point -> expected mechanism -> cheapest falsifying check. Label
expected benefit as a hypothesis. Separate paper reproduction from adapting
the method, since changed data or operating points alter the claim.

Check the verbatim FAQ for online/offline compute implications and prior
benchmark exposure. Do not turn "ideally CPU" into a fabricated absolute jury
rule, or claim an encoder is contamination-free without evidence.

Finish with a proposed experiment using `experiments/TEMPLATE.md`, including
controls, budget and decisive missing information. For actual execution use
the repository's experiment workflow and committed protocol. Literature review
alone does not authorize model downloads, paid jobs or a final test-split run.
