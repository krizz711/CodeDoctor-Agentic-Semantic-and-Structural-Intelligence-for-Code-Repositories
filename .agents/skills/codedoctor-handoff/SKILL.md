---
name: codedoctor-handoff
description: Record an evidence-backed CodeDoctor research milestone or prepare a reproducible handoff to another session or machine. Use when preserving experiment state, resuming a research cycle, or handing off blocked compute or validation work.
---

# CodeDoctor research handoff

Read `AGENTS.md`, `CLAUDE.md`, current Git status and the relevant experiment
records/addenda. Read only logs/artifacts needed to establish the current
state. Distinguish stored reports from commands executed in this session.

Update the applicable experiment record rather than creating a parallel global
history. Use `experiments/HANDOFF.md` only when its existing open-item list is
the right owner; preserve historical reproduction recipes and avoid rewriting
unrelated results.

Record:
- source branch/commit, dirty files, protocol/addenda and valid/void status;
- completed work with raw evidence paths, commands, actual exit results and
  relevant hashes; failed/skipped/unrun checks explicitly separated;
- decisions and limitations, including data exposure, operating point and
  degree of reviewer independence;
- next concrete action, required inputs, exact settled command/output paths,
  expected resources labelled as estimates, and stop/retry rule;
- unresolved user/team decisions or access that the next worker must obtain.

Do not transfer old machine/account facts to a new checkout. Do not relabel a
void run, promote a candidate to submitted status, or imply permission to spend
the test split. Inspect artifact metadata and controls when receiving results;
another agent's completion message is not execution evidence by itself.

A handoff document does not send a message, schedule a task, or start external
compute. Perform those actions only when the user requests them.
