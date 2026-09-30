"""Does dynamic ACTIVATION quantisation add error beyond weight-only W8?  Proxy: F2LLM-v2-160M
(same Qwen3 family; the 0.6B/1.7B do not fit the RAM free right now).  TUNE statements only,
compared with the stored GPU vectors.  Simulates the two runtimes' arithmetic in fp32 torch:
  llama.cpp Q8_0 : weights sym int8 per block of 32, activations sym int8 per block of 32
  OpenVINO default: weights asym int8 per output channel, activations int8 per group of 32 (sym here; unverified)
This is a simulation, not either runtime."""
import sys, time, json
from pathlib import Path
import numpy as np, torch
REPO = Path(r"C:/Users/Asus/manish/CodeDoctor"); sys.path.insert(0, str(REPO))
torch.set_num_threads(4)
from scripts.bench_query_cpu import _statements, MODELS
from app.retrieval.encode import CodeEncoder
N = int(sys.argv[1]) if len(sys.argv) > 1 else 40
rows, texts = _statements(REPO / "artifacts/apps", 100, 20260924)   # R2's seeded TUNE subset
rows, texts = rows[:N], texts[:N]
stored = np.load(REPO / "artifacts/apps/emb_f2llm160m_qtrain.npy").astype(np.float32)[rows]
name, rev, _ = MODELS["f2llm160m"]

def q_block_sym(w, g=32):
    sh = w.shape; w2 = w.reshape(-1, g)
    s = w2.abs().amax(1, keepdim=True).clamp_min(1e-12) / 127
    return (torch.round(w2 / s).clamp(-127, 127) * s).reshape(sh)
def q_row_asym(w):
    mn, mx = w.amin(1, keepdim=True), w.amax(1, keepdim=True)
    s = ((mx - mn) / 255).clamp_min(1e-12); zp = torch.round(-mn / s)
    return ((torch.clamp(torch.round(w / s) + zp, 0, 255) - zp) * s)

def run(cfg):
    enc = CodeEncoder(model_name=name, revision=rev, device="cpu", torch_dtype="float32")
    m = enc.model; am = m[0].auto_model
    hooks = []
    with torch.no_grad():
        for mod_name, mod in am.named_modules():
            if isinstance(mod, torch.nn.Linear):
                if cfg["w"] == "block32": mod.weight.copy_(q_block_sym(mod.weight))
                elif cfg["w"] == "row_asym": mod.weight.copy_(q_row_asym(mod.weight))
                if cfg["a"]:
                    hooks.append(mod.register_forward_pre_hook(lambda _m, inp: (q_block_sym(inp[0], 32),)))
        if cfg.get("emb"):
            e = am.embed_tokens.weight
            e.copy_(q_block_sym(e) if cfg["w"] == "block32" else q_row_asym(e))
    t = time.perf_counter()
    if cfg.get("drop_eos"):
        tok = m.tokenizer
        tok._tokenizer.post_processor = None  # negative control: no EOS appended
        tok.add_eos_token = False
        V = []
        for x in texts:
            V.append(m.encode([enc.query_prefix + x], normalize_embeddings=True)[0])
        V = np.asarray(V)
    else:
        V = np.vstack([enc.encode([x], mode="query") for x in texts])
    el = time.perf_counter() - t
    for h in hooks: h.remove()
    cos = (V * stored).sum(1) / np.linalg.norm(V, axis=1) / np.linalg.norm(stored, axis=1)
    from app.retrieval import encode as E; E._LOADED_MODELS.clear()
    return {"min": round(float(cos.min()), 5), "median": round(float(np.median(cos)), 5), "p10": round(float(np.quantile(cos, .1)), 5), "sec": round(el, 1)}

cfgs = {
  "fp32_control":           {"w": None, "a": False},
  "W8_block32_weightonly":  {"w": "block32", "a": False, "emb": True},
  "W8A8_block32(Q8_0-like)": {"w": "block32", "a": True, "emb": True},
  "W8_rowasym_weightonly":  {"w": "row_asym", "a": False, "emb": True},
  "W8rowasym_A8g32(OV-like)": {"w": "row_asym", "a": True, "emb": True},
  "NEG_drop_eos_fp32":      {"w": None, "a": False, "drop_eos": True},
}
res = {}
for k, c in cfgs.items():
    res[k] = run(c); print(k, res[k], flush=True)
json.dump({"n": N, "rows": rows, "results": res}, open("fakequant_160m.json", "w"), indent=1)
