#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.parser.chunker import chunk_source

SUPPORTED_EXTENSIONS = {".js", ".jsx", ".mjs", ".cjs", ".ts", ".tsx"}
IGNORED_DIRS = {".git", "node_modules", "dist", "build", "coverage", ".cache", ".idea", ".vscode"}


def get_git_sha(repo_path: Path) -> str:
    try:
        res = subprocess.run(
            ["git", "-C", str(repo_path), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True
        )
        return res.stdout.strip()
    except Exception:
        return "unknown_sha"


def run_span_fidelity(repo_path: Path) -> tuple[bool, str]:
    sha = get_git_sha(repo_path)
    repo_name = repo_path.name

    total_chunks = 0
    exact_matches = 0
    total_files = 0
    total_code_bytes = 0
    covered_bytes = 0

    for file_path in repo_path.rglob("*"):
        if not file_path.is_file():
            continue
        if any(ignored in file_path.parts for ignored in IGNORED_DIRS):
            continue
        if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        total_files += 1
        raw_bytes = file_path.read_bytes()
        try:
            source = raw_bytes.decode("utf-8")
        except UnicodeDecodeError:
            source = raw_bytes.decode("latin-1")

        chunks = chunk_source(source, str(file_path.relative_to(repo_path)))
        file_covered_intervals: list[tuple[int, int]] = []

        for chunk in chunks:
            total_chunks += 1
            sliced = raw_bytes[chunk.start_byte:chunk.end_byte]
            expected = chunk.content.encode("utf-8")
            if sliced == expected:
                exact_matches += 1
            else:
                if exact_matches == total_chunks - 1:
                    print(f"Mismatch in {file_path}:{chunk.start_byte}-{chunk.end_byte}")
                    print(f"Sliced len: {len(sliced)}, Expected len: {len(expected)}")

            if chunk.type != "FILE_SUMMARY":
                file_covered_intervals.append((chunk.start_byte, chunk.end_byte))

        # Calculate byte coverage of non-FILE_SUMMARY chunks
        file_covered_intervals.sort()
        merged_intervals: list[tuple[int, int]] = []
        for start, end in file_covered_intervals:
            if not merged_intervals or merged_intervals[-1][1] < start:
                merged_intervals.append((start, end))
            else:
                merged_intervals[-1] = (merged_intervals[-1][0], max(merged_intervals[-1][1], end))

        file_covered_bytes = sum(end - start for start, end in merged_intervals)
        covered_bytes += file_covered_bytes
        total_code_bytes += len(raw_bytes)

    if total_chunks == 0:
        return False, f"FAIL: No chunks found in {repo_path}"

    match_percent = (exact_matches / total_chunks) * 100.0
    coverage_percent = (covered_bytes / total_code_bytes * 100.0) if total_code_bytes else 0.0

    passed = (exact_matches == total_chunks)
    status = "PASS" if passed else "FAIL"
    msg = (
        f"{status}: {match_percent:.1f}% exact match  "
        f"({total_chunks} chunks across {total_files} files, repo={repo_name}@{sha})\n"
        f"Chunk coverage: {coverage_percent:.1f}% of total source bytes covered by AST chunks."
    )
    return passed, msg


def main():
    parser = argparse.ArgumentParser(description="Verify span fidelity across a repository.")
    parser.add_argument("repo_path", type=Path, nargs="?", default=Path("test_repos/commander"))
    args = parser.parse_args()

    passed, msg = run_span_fidelity(args.repo_path)
    print(msg)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
