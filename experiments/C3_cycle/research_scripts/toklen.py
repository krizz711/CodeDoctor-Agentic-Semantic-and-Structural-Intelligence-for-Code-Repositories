"""Read-only: token-length distribution of DEV train queries under the 1.7B tokenizer (offline cache)."""
import json, sys, os
os.environ["HF_HUB_OFFLINE"] = "1"
from pathlib import Path
import numpy as np
ROOT = Path(r"C:\Users\Asus\manish\CodeDoctor"); sys.path.insert(0, str(ROOT))
from transformers import AutoTokenizer
from app.pipeline.splits import make_development_split
from app.pipeline.query_analysis import analyze_query
from tokenizers import Tokenizer
_t = Tokenizer.from_file(str(Path.home()/".cache/huggingface/hub/models--codefuse-ai--F2LLM-v2-0.6B/snapshots/54b4e2dc74e01be7126d4cf5f016af6b21edc563/tokenizer.json"))
tok = lambda s: {"input_ids": _t.encode(s).ids}
A = ROOT / "artifacts" / "apps"
qids = json.loads((A / "ids_queries_train.json").read_text(encoding="utf-8"))
with open(A / "texts_queries_train.jsonl", encoding="utf-8") as h:
    qt = {r["id"]: r["text"] for r in map(json.loads, h)}
texts = [qt[q] for q in qids]
dev = make_development_split(texts, match_family=None, seed=20260919).dev
pre = "Instruct: Retrieve the most relevant code snippet for the given query.\nQuery: "
L = np.array([len(tok(pre + texts[i])["input_ids"]) for i in dev])
st = np.array([analyze_query(texts[i]).family.value == "competitive_stdin" for i in dev])
for name, m in (("all", np.ones_like(st)), ("stdin", st)):
    x = L[m]
    print(name, "n", m.sum(), "tokens p50/p90/p99/max", np.percentile(x, [50, 90, 99]).round(), x.max(), "mean", x.mean().round(1))
    for cap in (128, 256, 384, 512):
        kept = np.minimum(x, cap).sum() / x.sum()
        print(f"  cap {cap}: share truncated {(x > cap).mean():.3f}, tokens kept {kept:.3f}")
