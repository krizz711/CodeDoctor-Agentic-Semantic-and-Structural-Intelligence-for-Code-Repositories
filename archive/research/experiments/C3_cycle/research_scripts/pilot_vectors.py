"""Read-only: stored version-benchmark vectors (E5, APPS DEV lineages).

1. Reproduce E5 dense exec@1 / lineage@1 (0.6B known: 0.167 / 0.793) -> validates id order.
2. Cosine between versions of one lineage; query-score gap correct vs buggy.
3. P1 'vector reuse' approximation: every version of a lineage gets the vector of v1
   (first-indexed version) instead of its own. Does lineage@1 (right snippet first) change?
   Negative control: give each lineage the vector of a RANDOM other document -> must collapse.
"""
import json, sys
from pathlib import Path
import numpy as np

REPO = Path(r"C:\Users\Asus\manish\CodeDoctor")
V = REPO / "artifacts" / "versions_apps"
ids = json.loads((REPO / "artifacts" / "apps" / "ids_corpus.json").read_text(encoding="utf-8"))
lineage = json.loads((V / "lineage.json").read_text(encoding="utf-8"))
qrels = json.loads((V / "qrels.json").read_text(encoding="utf-8"))
replaced = {v["lineage"] for v in lineage.values()}
corpus_ids = [d for d in ids if d not in replaced] + list(lineage)
out = {}
for model in ["f2llm06b", "f2llm17b"]:
    C = np.load(V / f"emb_{model}_corpus.npy").astype(np.float32)
    Q = np.load(V / f"emb_{model}_queries.npy").astype(np.float32)
    assert C.shape[0] == len(corpus_ids), (C.shape, len(corpus_ids))
    C /= np.linalg.norm(C, axis=1, keepdims=True); Q /= np.linalg.norm(Q, axis=1, keepdims=True)
    pos = {d: i for i, d in enumerate(corpus_ids)}
    lin_of = [lineage.get(d, {}).get("lineage", d) for d in corpus_ids]
    members = {}
    for d, m in lineage.items():
        members.setdefault(m["lineage"], []).append(d)

    def evaluate(Cm):
        S = Q @ Cm.T
        e1 = l1 = 0
        for qi, (q, rel) in enumerate(qrels.items()):
            correct = next(iter(rel)); key = lineage[correct]["lineage"]
            top = int(np.argmax(S[qi]))
            e1 += corpus_ids[top] == correct
            l1 += lin_of[top] == key
        n = len(qrels)
        return e1 / n, l1 / n

    base = evaluate(C)
    cos_within, gaps = [], []
    S = Q @ C.T
    for qi, (q, rel) in enumerate(qrels.items()):
        correct = next(iter(rel)); key = lineage[correct]["lineage"]
        for d in members[key]:
            if d != correct:
                cos_within.append(float(C[pos[d]] @ C[pos[correct]]))
                gaps.append(float(S[qi, pos[correct]] - S[qi, pos[d]]))
    # margin to best non-lineage doc
    margins = []
    for qi, (q, rel) in enumerate(qrels.items()):
        key = lineage[next(iter(rel))]["lineage"]
        mask = np.array([l == key for l in lin_of])
        margins.append(float(S[qi][mask].max() - S[qi][~mask].max()))
    # reuse v1 vector for every version
    Cr = C.copy()
    for key, ds in members.items():
        v1 = sorted(ds, key=lambda d: lineage[d]["version"])[0]
        for d in ds:
            Cr[pos[d]] = C[pos[v1]]
    reuse = evaluate(Cr)
    # negative control: lineage vectors replaced with random other docs
    rng = np.random.default_rng(0)
    Cn = C.copy()
    base_rows = [i for i, d in enumerate(corpus_ids) if d not in lineage]
    for key, ds in members.items():
        r = C[rng.choice(base_rows)]
        for d in ds:
            Cn[pos[d]] = r
    neg = evaluate(Cn)
    out[model] = {
        "dense_exec@1": base[0], "dense_lineage@1": base[1],
        "cos_correct_vs_buggy_median": float(np.median(cos_within)),
        "cos_correct_vs_buggy_min": float(np.min(cos_within)),
        "cos_p5": float(np.percentile(cos_within, 5)),
        "query_gap_correct_minus_buggy_median": float(np.median(gaps)),
        "query_gap_abs_median": float(np.median(np.abs(gaps))),
        "lineage_margin_over_best_other_median": float(np.median(margins)),
        "lineage_margin_p5": float(np.percentile(margins, 5)),
        "reuse_v1_vector_lineage@1": reuse[1],
        "negative_control_random_vector_lineage@1": neg[1],
    }
print(json.dumps(out, indent=2))
(Path(__file__).parent / "pilot_vectors_result.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
