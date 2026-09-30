"""E2 (experiments/C3_cycle/PREREG.md): an OpenVINO int8 F2LLM-v2-1.7B query tower on CPU.

Runs inside two Kaggle CPU kernels built by scripts/build_e2_kernels.py.

    python scripts/e2_ov_kernel.py stage-a --data DIR --work DIR   # export, gate, timings, end-to-end
    python scripts/e2_ov_kernel.py stage-b --data DIR --work DIR --ir DIR --arm T1
    (internal) export / time / docrate subcommands, one fresh subprocess each

Every arm is timed in its own subprocess that loads only that arm's model, so its
peak RSS is its own; export runs in a separate process.  Timing blocks are
interleaved (REF06, T, REF / REF, T, REF06, ...) over the same queries.  The
kernel receives query and corpus texts and stored vectors only -- never qrels.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

M17 = ("codefuse-ai/F2LLM-v2-1.7B", "3766d46e7a68545ed6190c15330983f9b39ab718")
#: E2_SMOKE=1 checks the mechanics on Kaggle with the 160M model and a few
#: queries before the long run; nothing from a smoke run is a result.
SMOKE = os.environ.get("E2_SMOKE") == "1"
if SMOKE:
    M17 = ("codefuse-ai/F2LLM-v2-160M", "4ffe22c31406fc321b65a05b15563d38275ee51f")
STORED = "f2llm160m" if SMOKE else "f2llm17b"
LIMIT = int(os.environ.get("E2_LIMIT", "0") or 0)
M06 = ("codefuse-ai/F2LLM-v2-0.6B", "54b4e2dc74e01be7126d4cf5f016af6b21edc563")
EOS_ID = 151645
PINS = {"transformers": "4.57.6", "sentence-transformers": "5.0.0"}
STACK = ("torch", "transformers", "sentence-transformers", "openvino", "openvino-tokenizers", "nncf",
         "optimum", "optimum-intel", "numpy")
WARM_TEXT = "Given an array of integers, return the length of the longest increasing subsequence."
BLOCKS = 4


# -- helpers ------------------------------------------------------------------

def dump(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)),
                   encoding="utf-8")
    tmp.replace(path)


def peak_rss_gb() -> float:
    """This process's peak resident memory (Linux ru_maxrss is KiB; Windows peak working set)."""
    try:
        import resource

        return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 / 1e9, 3)
    except ImportError:
        import psutil

        info = psutil.Process().memory_info()
        return round(getattr(info, "peak_wset", info.rss) / 1e9, 3)


def versions() -> dict:
    from importlib.metadata import PackageNotFoundError, version

    out = {}
    for name in STACK:
        try:
            out[name] = version(name)
        except PackageNotFoundError:
            out[name] = None
    return out


def machine() -> dict:
    flags = set()
    model = ""
    try:
        for line in open("/proc/cpuinfo", encoding="utf-8"):
            if line.startswith("flags") and not flags:
                flags = set(line.split(":", 1)[1].split())
            if line.startswith("model name") and not model:
                model = line.split(":", 1)[1].strip()
    except OSError:
        pass
    import psutil

    wanted = ("avx2", "avx512f", "avx512_vnni", "avx_vnni", "avx512_bf16", "amx_int8", "amx_bf16", "fma")
    return {"cpu_model": model, "isa": {f: f in flags for f in wanted}, "platform": platform.platform(),
            "physical_cores": psutil.cpu_count(logical=False), "logical_cores": psutil.cpu_count(),
            "ram_gb": round(psutil.virtual_memory().total / 1e9, 1), "versions": versions()}


def check_pins() -> dict:
    got = versions()
    bad = {k: got.get(k) for k, v in PINS.items() if got.get(k) != v}
    return {"S6_pins_intact": not bad, "mismatch": bad, "nncf_present": got.get("nncf") is not None}


def cosines(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a / np.linalg.norm(a, axis=1, keepdims=True)
    b = b / np.linalg.norm(b, axis=1, keepdims=True)
    return (a * b).sum(1)


def cos_summary(c: np.ndarray) -> dict:
    return {"median": round(float(np.median(c)), 6), "min": round(float(c.min()), 6),
            "p10": round(float(np.percentile(c, 10)), 6)}


def run_sub(args: list[str], timeout: float) -> dict:
    """Run one subcommand in a fresh interpreter; its last JSON line is the result."""
    started = time.time()
    proc = subprocess.run([sys.executable, __file__, *args], capture_output=True, text=True,
                          timeout=timeout, cwd=str(REPO))
    lines = [line for line in proc.stdout.splitlines() if line.startswith("{")]
    if proc.returncode != 0 or not lines:
        return {"status": "failed", "rc": proc.returncode, "stderr_tail": proc.stderr[-4000:],
                "stdout_tail": proc.stdout[-2000:], "seconds": round(time.time() - started, 1)}
    result = json.loads(lines[-1])
    result.setdefault("status", "ok")
    return result


# -- subcommand: export ----------------------------------------------------------

def cmd_export(mode: str, out: str, snapshot: str) -> dict:
    """Export F2LLM-v2-1.7B to an OpenVINO sentence-transformers folder."""
    from optimum.intel import OVWeightQuantizationConfig
    from sentence_transformers import SentenceTransformer

    if mode == "C0":
        model_kwargs = {"load_in_8bit": False}
    elif mode == "T1":
        model_kwargs = {"quantization_config": OVWeightQuantizationConfig(bits=8, sym=True, group_size=32)}
    elif mode == "T1pc":
        model_kwargs = {"quantization_config": OVWeightQuantizationConfig(bits=8, sym=True, group_size=-1)}
    elif mode in ("T1b", "T1bpc"):
        # T1b keeps T1's grouping: per-channel once the group-32 export was rejected
        model_kwargs = {"quantization_config": OVWeightQuantizationConfig(
            bits=8, sym=True, group_size=-1 if mode == "T1bpc" else 32,
            ignored_scope={"patterns": [".*down_proj.*", ".*v_proj.*"]})}
    else:
        raise ValueError(mode)
    started = time.time()
    try:
        model = SentenceTransformer(snapshot, backend="openvino", device="cpu", model_kwargs=model_kwargs,
                                    tokenizer_kwargs={"add_eos_token": True, "model_max_length": 8192})
        model.save_pretrained(out)
    except Exception as error:  # the pre-registered substitution needs the reason
        return {"status": "export_error", "mode": mode, "error": f"{type(error).__name__}: {error}"[:2000]}
    ov_cfg = Path(out) / "openvino" / "openvino_config.json"
    config = json.loads(ov_cfg.read_text(encoding="utf-8")) if ov_cfg.exists() else None
    return {"status": "ok", "mode": mode, "seconds": round(time.time() - started, 1), "peak_rss_gb": peak_rss_gb(),
            "openvino_config": config, "layout": ir_layout(Path(out) / "openvino" / "openvino_model.xml"),
            "ir_bytes": sum(p.stat().st_size for p in Path(out).rglob("*") if p.is_file())}


def ir_layout(xml: Path) -> dict:
    """What the IR actually stores: int8 weight constants, and whether they are grouped."""
    try:
        import openvino as ov

        model = ov.Core().read_model(str(xml))
        types, grouped, per_channel = {}, 0, 0
        for op in model.get_ops():
            if op.get_type_name() != "Constant":
                continue
            et = op.get_output_element_type(0).get_type_name()
            types[et] = types.get(et, 0) + 1
            if et in ("i8", "u8"):
                shape = list(op.get_output_partial_shape(0).to_shape())
                if len(shape) == 3:
                    grouped += 1
                elif len(shape) == 2:
                    per_channel += 1
        return {"constant_types": types, "int8_grouped_3d": grouped, "int8_2d": per_channel}
    except Exception as error:
        return {"error": f"{type(error).__name__}: {error}"[:500]}


# -- subcommand: time one arm on one block of queries ----------------------------

def make_encoder(arm: str, ir: str | None):
    from app.retrieval.encode import FROZEN_CONFIG_PATH, CodeEncoder

    if arm == "REF":
        return CodeEncoder(model_name=M17[0], revision=M17[1], device="cpu", torch_dtype="float32",
                           tier="accuracy", config_path=FROZEN_CONFIG_PATH)
    if arm == "REF06":
        return CodeEncoder(model_name=M06[0], revision=M06[1], device="cpu", torch_dtype="float32",
                           tier="interactive", config_path=FROZEN_CONFIG_PATH)
    return CodeEncoder(model_name=M17[0], revision=M17[1], device="cpu", torch_dtype="float32",
                       tier="accuracy", config_path=FROZEN_CONFIG_PATH, backend="openvino", ov_path=ir)


def drop_terminal_eos(encoder) -> None:
    """NEG: strip the final token (the EOS) before the forward pass."""
    first = encoder.model._first_module()
    original = first.tokenize

    def tokenize(texts, *args, **kwargs):
        features = original(texts, *args, **kwargs)
        for key in ("input_ids", "attention_mask", "token_type_ids", "position_ids"):
            if key in features:
                features[key] = features[key][:, :-1]
        return features

    first.tokenize = tokenize


def ov_properties(encoder) -> dict:
    props = {}
    try:
        auto = encoder.model._first_module().auto_model
        compiled = getattr(auto, "compiled_model", None) or auto.request
        if hasattr(compiled, "get_compiled_model"):
            compiled = compiled.get_compiled_model()
        for name in ("INFERENCE_NUM_THREADS", "DYNAMIC_QUANTIZATION_GROUP_SIZE", "INFERENCE_PRECISION_HINT",
                     "KV_CACHE_PRECISION", "PERFORMANCE_HINT"):
            try:
                props[name] = str(compiled.get_property(name))
            except Exception as error:  # property names vary by version
                props[name] = f"n/a ({type(error).__name__})"
    except Exception as error:
        props["error"] = f"{type(error).__name__}: {error}"[:300]
    return props


def cmd_time(arm: str, texts_file: str, out_npy: str, ir: str | None, mode: str = "query") -> dict:
    import torch

    texts = json.loads(Path(texts_file).read_text(encoding="utf-8"))
    started = time.time()
    encoder = make_encoder("T" if arm.startswith(("T", "C0", "NEG")) else arm, ir)
    if arm == "NEG":
        drop_terminal_eos(encoder)
    encoder.encode([WARM_TEXT], mode=mode)  # load + warm-up, discarded
    load_seconds = round(time.time() - started, 2)
    tok = encoder.model.tokenizer
    prefix = encoder.query_prefix if mode == "query" else encoder.document_prefix
    ids = [tok.encode(prefix + t, add_special_tokens=True) for t in texts]
    s3 = all(x and x[-1] == EOS_ID and (len(x) < 2 or x[-2] != EOS_ID) for x in ids)
    latencies, vectors = [], []
    for text in texts:
        t = time.perf_counter()
        vectors.append(encoder.encode([text], mode=mode)[0])
        latencies.append(time.perf_counter() - t)
    np.save(out_npy, np.asarray(vectors, dtype=np.float32))
    return {"arm": arm, "n": len(texts), "load_seconds": load_seconds, "latencies": [round(x, 4) for x in latencies],
            "peak_rss_gb": peak_rss_gb(), "S3_single_terminal_eos": s3, "tokens_mean": float(np.mean([len(x) for x in ids])),
            "torch_threads_default": torch.get_num_threads(), "threads_set_by_encode": os.cpu_count(),
            "ov": ov_properties(encoder) if encoder.backend == "openvino" else None}


# -- orchestration -------------------------------------------------------------

def blocks(n: int) -> list[list[int]]:
    size = -(-n // BLOCKS)
    return [list(range(i, min(n, i + size))) for i in range(0, n, size)]


def interleaved(arms: dict[str, str | None], texts: list[str], work: Path, tag: str, timeout: float) -> dict:
    """Time every arm on the same texts, in alternating block order; vectors re-assembled per arm."""
    order_a = list(arms)
    lat = {a: [None] * len(texts) for a in arms}
    vec = {a: [None] * len(texts) for a in arms}
    meta = {a: [] for a in arms}
    for b, idx in enumerate(blocks(len(texts))):
        block_file = work / f"{tag}_block{b}.json"
        block_file.write_text(json.dumps([texts[i] for i in idx]), encoding="utf-8")
        for arm in (order_a if b % 2 == 0 else order_a[::-1]):
            out = work / f"{tag}_{arm}_b{b}.npy"
            args = ["time", "--arm", arm, "--texts", str(block_file), "--out", str(out)]
            if arms[arm]:
                args += ["--ir", arms[arm]]
            result = run_sub(args, timeout)
            if result.get("status") != "ok":
                return {"status": "failed", "arm": arm, "block": b, "detail": result}
            v = np.load(out)
            for k, i in enumerate(idx):
                lat[arm][i], vec[arm][i] = result["latencies"][k], v[k]
            meta[arm].append({k: result[k] for k in result if k not in ("latencies",)})
    return {"status": "ok", "latencies": lat, "vectors": {a: np.asarray(v) for a, v in vec.items()}, "meta": meta}


def ratio_ci(t: list[float], ref: list[float], seed: int = 7, reps: int = 2000) -> dict:
    t, ref = np.asarray(t), np.asarray(ref)
    rng = np.random.default_rng(seed)
    n = len(t)
    boots = [t[s].mean() / ref[s].mean() for s in (rng.integers(0, n, n) for _ in range(reps))]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"ratio": round(float(t.mean() / ref.mean()), 4), "ci95": [round(float(lo), 4), round(float(hi), 4)]}


def lat_summary(x: list[float]) -> dict:
    a = np.asarray(x)
    return {"mean": round(float(a.mean()), 3), "p50": round(float(np.percentile(a, 50)), 3),
            "p95": round(float(np.percentile(a, 95)), 3), "max": round(float(a.max()), 3)}


def efficiency(t: list[float], ref: list[float], rss_t: float, speed_bar: float) -> dict:
    r = ratio_ci(t, ref)
    p95_ratio = float(np.percentile(t, 95) / np.percentile(ref, 95))
    fast = r["ratio"] <= speed_bar and r["ci95"][1] < 1.0
    small = rss_t <= 4.5 and r["ratio"] <= 1.00 and p95_ratio <= 1.10
    return {**r, "p95_ratio": round(p95_ratio, 4), "peak_rss_gb": rss_t, "speed_bar": speed_bar,
            "fast_branch": fast, "small_branch": small, "pass": bool(fast or small)}


def execution_pass(data: Path, physical: int) -> dict:
    """Live candidate_v2 execution on the recorded pools of the timing statements (shared across arms)."""
    from app.pipeline.execution_rerank import ExecutionConfig, apply_execution_rerank, spawn_failures

    pools = json.loads((data / "tune100_pools.json").read_text(encoding="utf-8"))
    texts = json.loads((data / "tune100.json").read_text(encoding="utf-8"))
    query_of = dict(zip(texts["qids"], texts["texts"]))
    corpus = {}
    with open(data / "texts_corpus.jsonl", encoding="utf-8") as handle:
        for row in map(json.loads, handle):
            corpus[row["id"]] = row["text"]
    out = {}
    for fallback in (False, True):
        config = ExecutionConfig(enabled=True, max_candidates=20, timeout_seconds=2.0, numeric_tolerance=1e-6,
                                 call_style=True, deepen_to=50, skip_interactive=True, timeout_mode="cpu",
                                 workers=physical, verified_bonus=10.0, blank_line_fallback=fallback)
        secs = []
        for qid in texts["qids"][: LIMIT or None]:
            entry = pools[qid]
            t = time.perf_counter()
            apply_execution_rerank(query_of[qid], [corpus[d] for d in entry["pool"]], entry["scores"], config)
            secs.append(time.perf_counter() - t)
        out["blank_fallback_on" if fallback else "candidate_v2_as_registered"] = lat_summary(secs) | {"per_query": [round(s, 3) for s in secs]}
    out["spawn_failures"] = spawn_failures()
    out["workers"] = physical
    return out


def search_cost(data: Path) -> dict:
    out = {}
    for tag, dim in (("f2llm17b", 2048), ("f2llm06b", 1024)):
        corpus = np.load(data / f"emb_{tag}_corpus.npy").astype(np.float32)
        q = np.random.default_rng(0).standard_normal((100, dim)).astype(np.float32)
        secs = []
        for row in q:
            t = time.perf_counter()
            scores = corpus @ row
            np.argpartition(-scores, 50)[:50]
            secs.append(time.perf_counter() - t)
        out[tag] = lat_summary(secs)
    return out


def stage_a(data: Path, work: Path) -> dict:
    work.mkdir(parents=True, exist_ok=True)
    result: dict = {"stage": "A", "smoke": SMOKE, "limit": LIMIT, "model": M17, "machine": machine(),
                    "pins": check_pins()}
    dump(work / "e2_A.json", result)
    if not result["pins"]["S6_pins_intact"] or not result["pins"]["nncf_present"]:
        result["void"] = "S6: pinned transformers/sentence-transformers changed, or nncf missing"
        dump(work / "e2_A.json", result)
        return result
    from huggingface_hub import snapshot_download

    snap = snapshot_download(M17[0], revision=M17[1])
    snapshot_download(M06[0], revision=M06[1])
    tune = json.loads((data / "tune100.json").read_text(encoding="utf-8"))
    texts = tune["texts"][: LIMIT or None]
    stored17 = np.load(data / f"tune100_{STORED}.npy").astype(np.float32)[: len(texts)]
    timeout = 4 * 3600

    exports = {}
    for mode in ("C0", "T1"):
        exports[mode] = run_sub(["export", "--mode", mode, "--out", str(work / f"ir_{mode}"), "--snapshot", snap], timeout)
        result["exports"] = exports
        dump(work / "e2_A.json", result)
    if exports["T1"].get("status") == "export_error":
        # pre-registered substitution: grouped int8 rejected -> per-channel int8 sym
        exports["T1pc"] = run_sub(["export", "--mode", "T1pc", "--out", str(work / "ir_T1"), "--snapshot", snap], timeout)
        result["T1_substituted_by_per_channel"] = True
    t_ir = str(work / "ir_T1")

    # S1: the float32 OpenVINO export must reproduce the stored vectors
    c0 = run_sub(["time", "--arm", "C0", "--texts", str(_write(work / "tune_all.json", texts)),
                  "--out", str(work / "tune_C0.npy"), "--ir", str(work / "ir_C0")], timeout)
    result["C0"] = {k: v for k, v in c0.items() if k != "latencies"} | (
        {"latency": lat_summary(c0["latencies"]),
         "cos_vs_stored": cos_summary(cosines(np.load(work / "tune_C0.npy"), stored17))} if c0.get("status") == "ok" else {})
    ok_c0 = c0.get("status") == "ok" and result["C0"]["cos_vs_stored"]["median"] >= 0.9999 and result["C0"]["cos_vs_stored"]["min"] >= 0.999
    result["S1_C0_reproduces_stored"] = ok_c0
    shutil.rmtree(work / "ir_C0", ignore_errors=True)  # 7 GB; not needed again
    dump(work / "e2_A.json", result)

    # the gate: T1, then T1b, interleaved with REF and REF06 in the same session
    chosen = None
    for arm, ir in (("T1", t_ir), ("T1b", str(work / "ir_T1b"))):
        if arm == "T1b":
            mode = "T1bpc" if result.get("T1_substituted_by_per_channel") else "T1b"
            result["exports"]["T1b"] = run_sub(["export", "--mode", mode, "--out", ir, "--snapshot", snap], timeout)
            dump(work / "e2_A.json", result)
            if result["exports"]["T1b"].get("status") != "ok":
                break
        timed = interleaved({"REF06": None, arm: ir, "REF": None}, texts, work, f"gate_{arm}", timeout)
        if timed["status"] != "ok":
            result[f"gate_{arm}"] = timed
            break
        vec = timed["vectors"][arm]
        cos = cos_summary(cosines(vec, stored17))
        rss_t = max(m["peak_rss_gb"] for m in timed["meta"][arm])
        eff = efficiency(timed["latencies"][arm], timed["latencies"]["REF"], rss_t, speed_bar=0.77)
        fid = cos["median"] >= 0.995 and cos["min"] >= 0.98
        np.save(work / f"tune_{arm}.npy", vec)
        result[f"gate_{arm}"] = {
            "cos_vs_stored": cos, "fidelity_pass": fid, "efficiency": eff, "pass": bool(fid and eff["pass"]),
            "latency": {a: lat_summary(timed["latencies"][a]) for a in timed["latencies"]},
            "peak_rss_gb": {a: max(m["peak_rss_gb"] for m in timed["meta"][a]) for a in timed["meta"]},
            "load_seconds": {a: [m["load_seconds"] for m in timed["meta"][a]] for a in timed["meta"]},
            "S3_single_terminal_eos": all(m["S3_single_terminal_eos"] for ms in timed["meta"].values() for m in ms),
            "ov": timed["meta"][arm][0].get("ov"), "threads": {a: timed["meta"][a][0]["threads_set_by_encode"] for a in timed["meta"]},
            "REF_within_25pct_of_7.83s": abs(np.mean(timed["latencies"]["REF"]) / 7.832 - 1) <= 0.25,
            "raw_latencies": timed["latencies"],
        }
        dump(work / "e2_A.json", result)
        if result[f"gate_{arm}"]["pass"]:
            chosen = arm
            break
    result["chosen"] = chosen

    # negative control: the chosen int8 model without its terminal EOS
    neg_ir = str(work / f"ir_{chosen}") if chosen else t_ir
    neg = run_sub(["time", "--arm", "NEG", "--texts", str(work / "tune_all.json"), "--out", str(work / "tune_NEG.npy"),
                   "--ir", neg_ir], timeout)
    result["NEG"] = ({"cos_vs_stored": cos_summary(cosines(np.load(work / "tune_NEG.npy"), stored17))}
                     if neg.get("status") == "ok" else neg)
    result["NEG_fails_as_required"] = neg.get("status") == "ok" and result["NEG"]["cos_vs_stored"]["median"] < 0.9
    dump(work / "e2_A.json", result)

    # single-machine end-to-end parts (descriptive)
    import psutil

    result["search_cost"] = search_cost(data)
    result["execution_pass"] = execution_pass(data, psutil.cpu_count(logical=False) or 1)
    dump(work / "e2_A.json", result)
    docs = json.loads((data / "docs100.json").read_text(encoding="utf-8"))[: LIMIT or None]
    doc = run_sub(["time", "--arm", "REF", "--texts", str(_write(work / "docs.json", docs)), "--out", str(work / "docs_REF.npy"),
                   "--mode", "document"], timeout)
    result["doc_rate_1.7B_fp32"] = ({"latency": lat_summary(doc["latencies"]), "peak_rss_gb": doc["peak_rss_gb"]}
                                    if doc.get("status") == "ok" else doc)
    for p in work.glob("ir_T1b" if chosen != "T1b" else "ir_T1"):
        shutil.rmtree(p, ignore_errors=True)
    dump(work / "e2_A.json", result)
    return result


def _write(path: Path, obj) -> Path:
    path.write_text(json.dumps(obj), encoding="utf-8")
    return path


def stage_b(data: Path, work: Path, ir_src: Path, arm: str, stage_a_dir: Path) -> dict:
    work.mkdir(parents=True, exist_ok=True)
    result: dict = {"stage": "B", "smoke": SMOKE, "limit": LIMIT, "arm": arm, "model": M17, "machine": machine(),
                    "pins": check_pins()}
    dump(work / "e2_B.json", result)
    if not result["pins"]["S6_pins_intact"]:
        result["void"] = "S6"
        dump(work / "e2_B.json", result)
        return result
    from huggingface_hub import snapshot_download

    snapshot_download(M17[0], revision=M17[1])
    snapshot_download(M06[0], revision=M06[1])
    ir = work / "ir"
    if not ir.exists():
        shutil.copytree(ir_src, ir)
    timeout = 4 * 3600
    tune = json.loads((data / "tune100.json").read_text(encoding="utf-8"))
    tune["texts"] = tune["texts"][: LIMIT or None]
    rerun = run_sub(["time", "--arm", arm, "--texts", str(_write(work / "tune_all.json", tune["texts"])),
                     "--out", str(work / "tune_rerun.npy"), "--ir", str(ir)], timeout)
    prior = np.load(stage_a_dir / f"tune_{arm}.npy")
    c = cosines(np.load(work / "tune_rerun.npy"), prior) if rerun.get("status") == "ok" else None
    result["reproduces_stage_a"] = {"cos": cos_summary(c) if c is not None else rerun,
                                    "pass": c is not None and float(c.min()) >= 0.99999}
    dump(work / "e2_B.json", result)
    if not result["reproduces_stage_a"]["pass"]:
        result["void"] = "kernel B does not reproduce kernel A's vectors"
        dump(work / "e2_B.json", result)
        return result

    # adoption timing: 100 seeded DEV texts, interleaved, before the DEV encode
    timing = json.loads((data / "dev_timing100.json").read_text(encoding="utf-8"))
    timing["texts"] = timing["texts"][: LIMIT or None]
    timed = interleaved({"REF06": None, arm: str(ir), "REF": None}, timing["texts"], work, "adopt", timeout)
    if timed["status"] != "ok":
        result["adoption_timing"] = timed
        dump(work / "e2_B.json", result)
        return result
    rss = {a: max(m["peak_rss_gb"] for m in timed["meta"][a]) for a in timed["meta"]}
    result["adoption_timing"] = {
        "latency": {a: lat_summary(timed["latencies"][a]) for a in timed["latencies"]}, "peak_rss_gb": rss,
        "efficiency_vs_REF": efficiency(timed["latencies"][arm], timed["latencies"]["REF"], rss[arm], speed_bar=0.67),
        "ratio_vs_REF06": ratio_ci(timed["latencies"][arm], timed["latencies"]["REF06"]),
        "raw_latencies": timed["latencies"], "ov": timed["meta"][arm][0].get("ov"),
    }
    dump(work / "e2_B.json", result)

    # DEV + EXT-dev encode, batch 1, one subprocess; halve DEV by projection only
    dev = json.loads((data / "dev_texts.json").read_text(encoding="utf-8"))
    ext = json.loads((data / "extdev_texts.json").read_text(encoding="utf-8"))
    if LIMIT:
        dev = {"qids": dev["qids"][:LIMIT], "texts": dev["texts"][:LIMIT]}
        ext = {"qids": ext["qids"][:LIMIT], "texts": ext["texts"][:LIMIT]}
    head = run_sub(["time", "--arm", arm, "--texts", str(_write(work / "dev_head.json", dev["texts"][:50])),
                    "--out", str(work / "dev_head.npy"), "--ir", str(ir)], timeout)
    per_query = float(np.mean(head["latencies"])) if head.get("status") == "ok" else float("inf")
    projected_h = per_query * (len(dev["texts"]) + len(ext["texts"])) / 3600
    qids = list(dev["qids"])
    halved = projected_h > 6.0
    if halved:
        order = sorted(qids, key=lambda q: int(q[1:]))
        random.Random(20260926).shuffle(order)
        keep = set(order[:1250])
        qids = [q for q in dev["qids"] if q in keep]
    result["dev_plan"] = {"projected_hours": round(projected_h, 2), "halved": halved, "n": len(qids)}
    dump(work / "e2_B.json", result)
    text_of = dict(zip(dev["qids"], dev["texts"]))
    for name, ids, txts in (("dev", qids, [text_of[q] for q in qids]), ("extdev", ext["qids"], ext["texts"])):
        enc = run_sub(["time", "--arm", arm, "--texts", str(_write(work / f"{name}_texts.json", txts)),
                       "--out", str(work / f"{name}_{arm}.npy"), "--ir", str(ir)], 12 * 3600)
        result[f"encode_{name}"] = {"n": len(ids), "status": enc.get("status"),
                                    "latency": lat_summary(enc["latencies"]) if enc.get("status") == "ok" else enc}
        _write(work / f"{name}_{arm}_qids.json", ids)
        dump(work / "e2_B.json", result)
    shutil.rmtree(ir, ignore_errors=True)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("stage-a")
    a.add_argument("--data", type=Path, required=True)
    a.add_argument("--work", type=Path, required=True)
    b = sub.add_parser("stage-b")
    b.add_argument("--data", type=Path, required=True)
    b.add_argument("--work", type=Path, required=True)
    b.add_argument("--ir", type=Path, required=True)
    b.add_argument("--arm", required=True)
    b.add_argument("--stage-a-dir", type=Path, required=True)
    e = sub.add_parser("export")
    e.add_argument("--mode", required=True)
    e.add_argument("--out", required=True)
    e.add_argument("--snapshot", required=True)
    t = sub.add_parser("time")
    t.add_argument("--arm", required=True)
    t.add_argument("--texts", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--ir", default=None)
    t.add_argument("--mode", default="query")
    args = parser.parse_args(argv)
    if args.cmd in ("stage-a", "stage-b"):
        name = "e2_A.json" if args.cmd == "stage-a" else "e2_B.json"
        try:
            if args.cmd == "stage-a":
                done = stage_a(args.data, args.work)
            else:
                done = stage_b(args.data, args.work, args.ir, args.arm, args.stage_a_dir)
            print(json.dumps({"done": True, "chosen": done.get("chosen"), "void": done.get("void")}))
        except Exception:  # record the crash where the result would have been
            import traceback

            path = args.work / name
            partial = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
            partial["crash"] = traceback.format_exc()[-6000:]
            dump(path, partial)
            print(partial["crash"], file=sys.stderr)
            return 1
    elif args.cmd == "export":
        print(json.dumps(cmd_export(args.mode, args.out, args.snapshot)))
    elif args.cmd == "time":
        print(json.dumps(cmd_time(args.arm, args.texts, args.out, args.ir, args.mode)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
