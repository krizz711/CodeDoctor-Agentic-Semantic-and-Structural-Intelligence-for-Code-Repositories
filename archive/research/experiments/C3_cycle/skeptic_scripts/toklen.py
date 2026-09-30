"""Token lengths of TUNE statements under the frozen query recipe (prefix + text + EOS)."""
import gzip, json, sys
from pathlib import Path
import numpy as np
REPO = Path(r"C:/Users/Asus/manish/CodeDoctor")
sys.path.insert(0, str(REPO))
from transformers import AutoTokenizer
from scripts.dev_eval import load_jsonl_texts
from app.retrieval.encode import APPS_QUERY_PREFIX
snap = next(Path(r"C:/Users/Asus/.cache/huggingface/hub/models--codefuse-ai--F2LLM-v2-0.6B/snapshots").glob("54b4*"))
tok = AutoTokenizer.from_pretrained(str(snap), add_eos_token=True)
ids, texts = load_jsonl_texts(REPO / "artifacts/apps/texts_queries_train.jsonl")
with gzip.open(REPO / "data/outcomes/f2llm17b_all_tune.jsonl.gz", "rt", encoding="utf-8") as f:
    tune_q = [json.loads(l).get("qid") for l in f][1:]
pos = {q: i for i, q in enumerate(ids)}
rows = [pos[q] for q in tune_q]
enc = [tok.encode(APPS_QUERY_PREFIX + texts[i], add_special_tokens=True) for i in rows]
lens = np.array([len(e) for e in enc])
print("eos appended on all:", all(e[-1] == tok.eos_token_id for e in enc), "eos id", tok.eos_token_id, tok.eos_token)
print("TUNE n", len(lens), "mean", lens.mean().round(1), "median", np.median(lens), "p95", np.percentile(lens, 95), "max", lens.max(), "sum", lens.sum())
print("sum of squares ratio (attention weight) mean(L^2)/mean(L)^2:", round(float((lens**2).mean() / lens.mean()**2), 2))
# the literal-string route used by llama.cpp users: does "<|im_end|>" text tokenize to the EOS id?
t2 = AutoTokenizer.from_pretrained(str(snap))
print("no-flag tokenizer appends eos:", t2.encode("abc")[-1] == t2.eos_token_id, "| literal '<|im_end|>' ->", t2.encode("abc<|im_end|>")[-2:], "| '<|endoftext|>' ->", t2.encode("abc<|endoftext|>")[-1:])
