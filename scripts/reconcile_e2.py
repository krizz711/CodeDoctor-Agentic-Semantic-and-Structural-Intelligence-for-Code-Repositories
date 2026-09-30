"""Adjudicate a saved C3-E2 stage-A result without running or changing it.

This does not download data, certify its provenance, or adopt an encoder.
The caller must obtain version 2 from the notebook identified in HANDOFF A4.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

MODEL = ["codefuse-ai/F2LLM-v2-1.7B", "3766d46e7a68545ed6190c15330983f9b39ab718"]


def assess(result: dict, *, attempt: int) -> dict:
    if attempt not in (1, 2):
        raise ValueError("C3-E2 permits only two attempts")
    missing: list[str] = []
    failed: list[str] = []

    def check(name: str, value) -> None:
        if value is None:
            missing.append(name)
        elif value is not True:
            failed.append(name)

    def threshold(name: str, value, predicate) -> None:
        if value is None:
            missing.append(name)
        elif not isinstance(value, (float, int)) or isinstance(value, bool) or not math.isfinite(value):
            failed.append(name)
        else:
            check(name, predicate(value))

    check("stage A", result["stage"] == "A" if "stage" in result else None)
    check("pinned model", result["model"] == MODEL if "model" in result else None)
    check("not smoke", result["smoke"] is False if "smoke" in result else None)
    check("no sample limit", result["limit"] == 0 if "limit" in result else None)
    if result.get("void"):
        failed.append(f"driver void: {result['void']}")
    check("S1 reported", result.get("S1_C0_reproduces_stored"))
    c0 = result.get("C0", {}).get("cos_vs_stored", {})
    threshold("S1 median", c0.get("median"), lambda x: x >= .9999)
    threshold("S1 minimum", c0.get("min"), lambda x: x >= .999)
    check("S6 pins", result.get("pins", {}).get("S6_pins_intact"))
    check("S4 nncf installed", result.get("pins", {}).get("nncf_present"))
    check("NEG reported", result.get("NEG_fails_as_required"))
    threshold("NEG cosine", result.get("NEG", {}).get("cos_vs_stored", {}).get("median"), lambda x: x < .9)
    arms = {}
    for arm in ("T1", "T1b"):
        gate = result.get(f"gate_{arm}")
        if gate is None:
            if arm == "T1" or not arms.get("T1", {}).get("passes"):
                missing.append(f"gate_{arm}")
            continue
        check(f"{arm} S2 reported", gate.get("REF_within_25pct_of_7.83s"))
        check(f"{arm} S3 EOS", gate.get("S3_single_terminal_eos"))
        raw = gate.get("raw_latencies", {})
        ref = raw.get("REF", [])
        if len(ref) != 100 or any(not isinstance(x, (float, int)) or not math.isfinite(x) or x <= 0 for x in ref):
            missing.append(f"{arm} 100 finite REF timings")
        else:
            # The committed driver uses 7.832 as the unrounded R2 reference.
            check(f"{arm} S2 recomputed", abs(sum(ref) / len(ref) / 7.832 - 1) <= .25)
        for key in (arm, "REF06"):
            vals = raw.get(key, [])
            if len(vals) != 100 or any(not isinstance(x, (float, int)) or not math.isfinite(x) or x <= 0 for x in vals):
                missing.append(f"{arm} 100 finite {key} timings")
        export_key = "T1pc" if arm == "T1" and result.get("T1_substituted_by_per_channel") else arm
        export = result.get("exports", {}).get(export_key, {})
        check(f"{arm} S4 export", export.get("status") == "ok" if export else None)
        types = export.get("layout", {}).get("constant_types", {})
        check(f"{arm} S4 int8 constants", (types.get("i8", 0) + types.get("u8", 0)) > 0 if types else None)
        cos = gate.get("cos_vs_stored", {})
        eff = gate.get("efficiency", {})
        needed = (cos.get("median"), cos.get("min"), eff.get("ratio"), eff.get("p95_ratio"), eff.get("peak_rss_gb"))
        ci = eff.get("ci95", [])
        if (any(not isinstance(v, (float, int)) or not math.isfinite(v) for v in needed)
                or len(ci) != 2 or any(not isinstance(v, (float, int)) or not math.isfinite(v) for v in ci)):
            missing.append(f"{arm} fidelity/efficiency measurements")
            continue
        fidelity = cos["median"] >= .995 and cos["min"] >= .98
        efficiency = ((eff["ratio"] <= .77 and ci[1] < 1)
                      or (eff["peak_rss_gb"] <= 4.5 and eff["ratio"] <= 1 and eff["p95_ratio"] <= 1.1))
        passes = fidelity and efficiency
        check(f"{arm} gate agrees with measurements", gate.get("pass") is passes)
        arms[arm] = {"passes": passes, "fidelity": fidelity, "efficiency": efficiency,
                     "cosine": cos, "timing": eff, "latency": gate.get("latency"),
                     "peak_rss_gb": gate.get("peak_rss_gb")}
    chosen = next((a for a in ("T1", "T1b") if arms.get(a, {}).get("passes")), None)
    if "chosen" not in result:
        missing.append("chosen")
    else:
        check("chosen agrees with fixed arm order", result["chosen"] == chosen)
    if failed:
        status = "void"
        action = "stop E2; no more repeats" if attempt == 2 else "one documented repeat permitted"
    elif missing:
        status, action = "incomplete", "recover complete result/provenance; no adoption or new experiment decision"
    elif chosen:
        status, action = "gate_passed", "continue original E2 stage B; defer Q8_0"
    else:
        status, action = "negative", "stop E2; Q8_0 preregistration may proceed"
    return {"status": status, "attempt": attempt, "next_action": action,
            "failed_controls": failed, "missing_evidence": missing, "chosen": chosen,
            "arms": arms, "machine": result.get("machine"),
            "search_cost": result.get("search_cost"), "execution_pass": result.get("execution_pass"),
            "doc_rate_1.7B_fp32": result.get("doc_rate_1.7B_fp32"),
            "adoption": False, "provenance": "caller must independently verify notebook/version"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    parser.add_argument("--attempt", type=int, required=True, choices=(1, 2))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    raw = args.result.read_bytes()
    original = Path(__file__).resolve().parents[1] / "experiments/C3_cycle/E2_runs/e2_A_run1_VOID.json"
    if args.attempt == 2 and raw == original.read_bytes():
        raise ValueError("This is the recorded run-1 artifact, not evidence for run 2")
    report = assess(json.loads(raw), attempt=args.attempt)
    report["source_sha256"] = hashlib.sha256(raw).hexdigest()
    report["source_path"] = str(args.result)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as f:
        json.dump(report, f, indent=2, allow_nan=False)
        f.write("\n")
    print(json.dumps({k: report[k] for k in ("status", "next_action", "failed_controls", "missing_evidence")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
