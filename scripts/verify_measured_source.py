"""Check that the source a recorded run measured is byte-identical to a commit.

    python scripts/verify_measured_source.py --environment results/final/p0/environment.json --commit HEAD

Every stage of ``scripts/submission_run.py`` records the sha256 of each source, config and
requirements file it ran (``source_sha256``) and of ``config/submission.yaml``. This reads the
same paths as raw blobs from ``--commit`` (never the working tree, whose line endings vary) and
reports any file that differs or is missing. Exit status 1 means the run did not measure the
code at that commit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def blob_sha256(commit: str, path: str, repo: Path = ROOT) -> str | None:
    result = subprocess.run(["git", "-C", str(repo), "cat-file", "blob", f"{commit}:{path}"], capture_output=True)
    return hashlib.sha256(result.stdout).hexdigest() if result.returncode == 0 else None


def verify(environment: dict, commit: str, repo: Path = ROOT) -> dict:
    measured = environment.get("source_sha256") or {}
    if not measured:
        raise ValueError("the environment record has no source_sha256")
    differing, missing = [], []
    for path, recorded in sorted(measured.items()):
        actual = blob_sha256(commit, path, repo)
        if actual is None:
            missing.append(path)
        elif actual != recorded:
            differing.append(path)
    config_ok = blob_sha256(commit, "config/submission.yaml", repo) == environment.get("configuration", {}).get("sha256")
    resolved = subprocess.run(["git", "-C", str(repo), "rev-parse", commit], capture_output=True, text=True,
                              check=True).stdout.strip()
    return {"commit": resolved, "recorded_commit": environment.get("commit"), "files_checked": len(measured),
            "differing": differing, "missing": missing, "configuration_matches": config_ok,
            "identical": not differing and not missing and config_ok}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--environment", type=Path, required=True, help="a stage's environment.json")
    parser.add_argument("--commit", default="HEAD")
    args = parser.parse_args(argv)
    report = verify(json.loads(args.environment.read_text(encoding="utf-8")), args.commit)
    print(json.dumps(report, indent=2))
    return 0 if report["identical"] else 1


if __name__ == "__main__":
    sys.exit(main())
