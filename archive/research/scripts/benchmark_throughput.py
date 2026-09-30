from __future__ import annotations

import os
import sys
import time
from pathlib import Path
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.evaluation_harness import DevelopmentHarness
from app.retrieval.encode import CodeEncoder


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Benchmark throughput on CPU and GPU (Task 3.1 & 3.5).")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"], help="Compute device")
    args = parser.parse_args()

    print("=== Section 1 & 3.1 Throughput Diagnosis & Benchmark ===")
    cores = os.cpu_count() or 1
    default_threads = torch.get_num_threads()
    cuda_avail = torch.cuda.is_available()
    print(f"System Cores: {cores}, Default PyTorch Threads: {default_threads}")
    print(f"CUDA Available: {cuda_avail}")
    if cuda_avail:
        print(f"CUDA Device: {torch.cuda.get_device_name(0)}")

    harness = DevelopmentHarness(firewall_enabled=True)
    corpus = harness.get_corpus()
    corpus_sample = list(corpus.values())[:64]

    # CPU Benchmark (always run to verify B5 latency and CPU baseline)
    print("\n--- CPU Encoder Benchmark ---")
    cpu_encoder = CodeEncoder(device="cpu", max_seq_length=512)
    model = cpu_encoder.model
    cpu_param_dtype = next(model.parameters()).dtype
    print(f"  Model Parameter Dtype: {cpu_param_dtype}")
    print(f"  SentenceTransformer max_seq_length: {getattr(model, 'max_seq_length', 512)}")
    print(f"  Active PyTorch Threads: {torch.get_num_threads()}")

    # Analyze token distribution of the 64 real documents
    tok_counts = []
    for doc_text in corpus_sample:
        prefixed = cpu_encoder.get_prompt_text(doc_text, mode="document")
        tokens = model.tokenize([prefixed])["input_ids"][0]
        tok_counts.append(len(tokens))

    print("\nToken Count Distribution (64 APPS Docs with max_seq_length=512):")
    print(f"  Min: {min(tok_counts)}, Max: {max(tok_counts)}, Mean: {np.mean(tok_counts):.1f}, Median: {np.median(tok_counts):.1f}")
    print(f"  Truncated (>=512): {sum(1 for t in tok_counts if t >= 512)} / 64 docs")

    # Section 3.5: Single function edit / single chunk re-index latency
    print("\n=== Section 3.5: B5 Single-Chunk Re-index Latency (Target: < 1.0s on CPU) ===")
    single_func = "function calculateTotal(items) { return items.reduce((a, b) => a + b.price, 0); }"
    t_single_start = time.perf_counter()
    _ = cpu_encoder.encode(single_func, mode="document")
    t_single = time.perf_counter() - t_single_start
    print(f"Single function edit -> re-index wall clock = {t_single:.4f} sec ({t_single*1000:.1f} ms)")
    if t_single < 1.0:
        print("PASS: B5 single-chunk re-index is sub-second on CPU!")
    else:
        print("FAIL: B5 single-chunk re-index exceeded 1.0s!")

    # Benchmark 64-document batch on CPU
    print("\nEncoding 64 real documents on CPU (batch_size=64)...")
    t0 = time.perf_counter()
    _ = cpu_encoder.encode(corpus_sample, mode="document", batch_size=64)
    dt_cpu = time.perf_counter() - t0
    sec_per_doc_cpu = dt_cpu / len(corpus_sample)
    print(f"CPU Total time: {dt_cpu:.2f}s ({sec_per_doc_cpu:.4f} sec/doc)")
    print(f"Full corpus (8765 docs) CPU projection: {sec_per_doc_cpu * 8765 / 60.0:.1f} minutes")

    # GPU Benchmark if CUDA available
    if cuda_avail and args.device != "cpu":
        print("\n--- GPU Encoder Benchmark (NVIDIA CUDA) ---")
        gpu_encoder = CodeEncoder(device="cuda", max_seq_length=512)
        gpu_param_dtype = next(gpu_encoder.model.parameters()).dtype
        print(f"  GPU Parameter Dtype: {gpu_param_dtype}")
        # Warmup GPU
        _ = gpu_encoder.encode("function init() {}", mode="document")
        torch.cuda.synchronize()

        print("Encoding 64 real documents on GPU (batch_size=64)...")
        t0 = time.perf_counter()
        _ = gpu_encoder.encode(corpus_sample, mode="document", batch_size=64)
        torch.cuda.synchronize()
        dt_gpu_64 = time.perf_counter() - t0
        sec_per_doc_gpu_64 = dt_gpu_64 / len(corpus_sample)

        print("Encoding 64 real documents on GPU (batch_size=256)...")
        t0 = time.perf_counter()
        _ = gpu_encoder.encode(corpus_sample, mode="document", batch_size=256)
        torch.cuda.synchronize()
        dt_gpu_256 = time.perf_counter() - t0
        sec_per_doc_gpu_256 = dt_gpu_256 / len(corpus_sample)

        speedup_container_cpu = sec_per_doc_cpu / sec_per_doc_gpu_256
        # Reference baseline: Local 12-core host workstation (175.52s / 64 docs = 2.7425 sec/doc)
        local_12core_sec_per_doc = 2.7425
        speedup_12core_cpu = local_12core_sec_per_doc / sec_per_doc_gpu_256

        print(f"\n[GPU Throughput & Dual-Baseline Comparison]")
        print(f"GPU batch 64:  {dt_gpu_64:.2f}s ({sec_per_doc_gpu_64:.4f} sec/doc)")
        print(f"GPU batch 256: {dt_gpu_256:.2f}s ({sec_per_doc_gpu_256:.4f} sec/doc)")
        print(f"Speedup vs Current Container CPU (2 vCPUs, {dt_cpu:.1f}s): {speedup_container_cpu:.2f}x")
        print(f"Speedup vs Workstation CPU (12 cores, 175.5s):        {speedup_12core_cpu:.2f}x")
        print(f"Full corpus (8765 docs) GPU projection: {sec_per_doc_gpu_256 * 8765 / 60.0:.1f} minutes")


if __name__ == "__main__":
    main()
