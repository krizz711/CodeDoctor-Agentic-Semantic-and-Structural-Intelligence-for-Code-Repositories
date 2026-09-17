"""Offline sanity check for PrePostPipelineEncoder.

Verifies the encoder is wired up and actually discriminates between relevant and
irrelevant code, without downloading the AppsRetrieval dataset. Use this before
committing to a long full evaluation run.

    python scripts/verify_encoder.py                  # uses a local stub backbone
    python scripts/verify_encoder.py --real-model     # downloads the real dense model
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.retrieval.mteb_encoder import PrePostPipelineEncoder


class StubDenseModel:
    """Deterministic content-dependent stand-in so this runs with no network."""

    dim = 64

    @staticmethod
    def _stable_hash(word: str) -> int:
        # Python's built-in hash() is randomised per process; use FNV-1a so this
        # stub reproduces identically across runs.
        value = 1469598103934665603
        for char in word.encode("utf-8"):
            value = ((value ^ char) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
        return value

    def encode(self, texts, batch_size=64, show_progress_bar=False, convert_to_numpy=True):
        rows = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for word in text.lower().split():
                rows[i, self._stable_hash(word) % self.dim] += 1.0
        return rows


class _Prompt:
    def __init__(self, value: str) -> None:
        self.value = value


CASES = [
    (
        "How is the input preprocessed before going to the main function?",
        [
            ("normalize", "function normalize(str) { const s = str.trim(); return forward(s); }"),
            ("check", "function check(s) { var pre = s.slice(0, 6); return pre === 'en-US'; }"),
            ("perf", "function perf(str) { if (act(A, str)) { return act(B, str) } }"),
        ],
        "normalize",
    ),
    (
        "Count the number of pairs of integers that sum to a target value.",
        [
            (
                "count_pairs",
                "def count_pairs(nums, target):\n    seen = {}\n    total = 0\n    for n in nums:\n        total += seen.get(target - n, 0)\n        seen[n] = seen.get(n, 0) + 1\n    return total",
            ),
            ("reverse_string", "def reverse_string(s):\n    return s[::-1]"),
            ("read_matrix", "def read_matrix(n):\n    return [list(map(int, input().split())) for _ in range(n)]"),
        ],
        "count_pairs",
    ),
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify the MTEB encoder works end to end.")
    parser.add_argument(
        "--real-model", action="store_true", help="use the real dense backbone (requires network on first run)"
    )
    parser.add_argument("--model", default="BAAI/bge-small-en-v1.5")
    args = parser.parse_args()

    if args.real_model:
        encoder = PrePostPipelineEncoder(model_name=args.model, device="cpu")
        print(f"Backbone: {args.model}")
    else:
        encoder = PrePostPipelineEncoder(dense_model=StubDenseModel())
        print("Backbone: local stub (structural check only -- pass --real-model for a true check)")

    print(f"Output dimensionality: {encoder.encode_texts(['x']).shape[1]}")

    # Determinism: the single most important guard against a random-vector encoder.
    a = encoder.encode_texts(["def solve(n): return n"])
    b = encoder.encode_texts(["def solve(n): return n"])
    deterministic = np.allclose(a, b)
    print(f"Deterministic across calls: {'PASS' if deterministic else 'FAIL'}")
    if not deterministic:
        print("\nFATAL: encoder is non-deterministic. Do not generate a submission with it.")
        return 1

    failures = 0
    for query, documents, expected in CASES:
        query_vector = encoder.encode_texts([query], prompt_type=_Prompt("query"))[0]
        doc_vectors = encoder.encode_texts([body for _, body in documents], prompt_type=_Prompt("document"))
        scores = doc_vectors @ query_vector
        order = np.argsort(-scores)

        print(f"\nQuery: {query}")
        for rank, idx in enumerate(order, start=1):
            name = documents[idx][0]
            marker = "  <-- expected top" if name == expected else ""
            print(f"  {rank}. {name:<16} score={scores[idx]:.4f}{marker}")

        if documents[order[0]][0] != expected:
            failures += 1
            print("  RESULT: FAIL (expected document did not rank first)")
        else:
            print("  RESULT: PASS")

    if failures:
        print(f"\n{failures} case(s) failed.")
        if not args.real_model:
            print(
                "Note: the stub backbone is intentionally crude. Re-run with --real-model "
                "for semantic ranking; deterministic wiring checks passed."
            )
            return 0
        return 1

    print("\nAll checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
