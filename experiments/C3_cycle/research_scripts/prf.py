"""TUNE-only: dense vector PRF (Rocchio) at 1.7B and 0.6B, dense NDCG@10 and recall@20/50."""
import json, sys
import numpy as np
from pathlib import Path
REPO = Path(r"C:\Users\Asus\manish\CodeDoctor"); sys.path.insert(0, str(REPO))
from app.pipeline.splits import make_development_split
from scripts.dev_eval import load_jsonl_texts
art=REPO/"artifacts/apps"
cids,_=load_jsonl_texts(art/"texts_corpus.jsonl"); qids,qt=load_jsonl_texts(art/"texts_queries_train.jsonl")
qrels=json.loads((art/"qrels_train.json").read_text(encoding="utf-8")); col={d:i for i,d in enumerate(cids)}
gc=np.array([col[next(iter(qrels[q]))] for q in qids])
split=make_development_split(qt, match_family=None, seed=20260919); rows=np.asarray(split.tune)
def ev(S,g):
    gs=S[np.arange(len(g)),g]; rank=(S>gs[:,None]).sum(1)+1
    return np.where(rank<=10,1/np.log2(rank+1),0).mean(), (rank<=20).mean(), (rank<=50).mean(), (rank==1).mean()
for m in ["f2llm17b","f2llm06b"]:
    D=np.load(art/f"emb_{m}_corpus.npy").astype(np.float32); Q=np.load(art/f"emb_{m}_qtrain.npy").astype(np.float32)[rows]
    g=gc[rows]; S=Q@D.T
    print(m,"base ndcg/R20/R50/R1", np.round(ev(S,g),5))
    top=np.argsort(-S,1)[:,:10]
    for k in [1,3,5]:
        for b in [0.1,0.3,0.5]:
            fb=D[top[:,:k]].mean(1); Q2=Q+b*fb; Q2/=np.linalg.norm(Q2,axis=1,keepdims=True)
            print(f"  k={k} beta={b}", np.round(ev(Q2@D.T,g),5))
