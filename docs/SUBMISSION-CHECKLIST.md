# Submission checklist

Theme 01 asks for three things. Two are done and in the repository; the third
is a GitHub release, which is a manual step on purpose.

| deliverable | state |
|---|---|
| MTEB results JSON from the test split | **done** — [`appsretrieval_results.json`](../appsretrieval_results.json), NDCG@10 = 0.96558 |
| Deck | **done** — [`CodeDoctor-PRISM-Theme01.pptx`](CodeDoctor-PRISM-Theme01.pptx), 13 slides |
| Repository with runnable instructions | **done** — [`README.md`](../README.md) §1 and §6 |
| Attached to a GitHub release, tagged | **outstanding** — steps below |

---

## 1. The release

> **Check the tag against the portal before pushing it.** The official
> `theme1_guidelines.pdf` does **not** name a tag. It says only "make a GitHub
> release and upload the generated file" and links GitHub's own release guide.
> The string `PRISM_GENAI_HACKATHON_Y2026` comes from our internal
> `THEME1-BUILD-PLAN-v3.pdf`, p11 — it may have been taken from a submission
> portal we cannot see from here, or it may have been invented during
> planning. A release and a tag can both be deleted and recreated, so this is
> recoverable, but it is worth thirty seconds of checking because the tag is
> how judges find the submission.

Tag (unverified, see above): **`PRISM_GENAI_HACKATHON_Y2026`**

What the guidelines *do* require, verbatim: a **CSV of the responses from the
test split**, and the **MTEB JSON** attached to a release. Both are in the
list below.

Attach these four files. All are committed at the repository root, so they can
be dragged straight from a clone — or downloaded from the tagged tree on
GitHub:

| file | what it is |
|---|---|
| `appsretrieval_results.json` | the scored artifact, MTEB's own output, 8.3 KB |
| `submission_predictions.csv` | ranked responses, 3,765 queries × top 10, 884 KB |
| `submission_metrics.csv` | the flat metric table, 1 KB |
| `execution_audit.json` | what the third pass did: 73,520 candidates executed, 3,385 verified |

```bash
git tag -a PRISM_GENAI_HACKATHON_Y2026 -m "Samsung PRISM GenAI Hackathon 2026 - Theme 01 submission"
git push origin PRISM_GENAI_HACKATHON_Y2026
```

Then create the release from that tag in the GitHub UI and attach the four
files. A tag and a release can both be deleted and recreated if the name turns
out to be wrong, so this is recoverable — but it is worth checking the exact
string against the official brief before pushing, since the tag is how judges
find the submission.

## 2. Making the Kaggle notebook public

The reproduction notebook is
`munalsingh/codedoctor-appsretrieval-reproduction`. It is private, and it
holds the saved cell outputs of a full end-to-end run.

> **Flip it public from the Kaggle UI** — notebook → *Settings* → *Sharing* →
> *Public*.
>
> Do **not** flip it by editing `kernel-metadata.json` and re-pushing.
> `kaggle kernels push` creates a new *version* and re-runs the notebook from
> scratch, which discards the saved outputs and costs about three and a half
> hours of GPU time to get back. The metadata change is not worth a re-run.

Once it is public, paste the link into the deck's closing slide and into
[`README.md`](../README.md) §6, next to the generator table.

## 3. Worth re-checking on the day

- `pytest -q` — the full suite including the two weight-loading tests.
- `python scripts/validate_submission.py appsretrieval_results.json` — should
  print `Valid retrieval result`.
- The headline number appears in six places and they must agree:
  `appsretrieval_results.json`, `submission_metrics.csv`, `README.md`,
  `docs/SUBMISSION-DECK.md`, `docs/RETRIEVAL-PIPELINE.md` and the built
  `.pptx`. All six currently read **0.96558** — verified, not assumed:

  ```bash
  grep -rl 0.96558 appsretrieval_results.json submission_metrics.csv README.md docs/
  ```
