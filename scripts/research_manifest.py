"""Record a local environment and file hashes beside an existing research run.

This does not run experiments or verify how an earlier result was produced.
Run immediately after a run, before changing its checkout or environment.
"""

import argparse
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import stat
import subprocess
import tempfile


class ManifestError(Exception):
    """A manifest cannot be collected or safely written."""


def _identity(file_stat):
    """Which file it is: stable across a path lookup and an open handle."""
    return (file_stat.st_dev, file_stat.st_ino)


def _fingerprint(file_stat):
    """The file's state, compared only between two reads of the same open handle.

    A path lookup is not comparable: on NTFS a just-written file's directory
    entry can report an older size or timestamp than its open handle until it
    is flushed, which made a freshly written multi-megabyte file look "changed".
    """
    return (
        file_stat.st_dev, file_stat.st_ino, file_stat.st_size,
        file_stat.st_mtime_ns, file_stat.st_ctime_ns,
    )


def hash_file(path):
    """Hash one regular file without loading it into memory."""
    path = Path(path).resolve()
    try:
        initial = path.stat()
        if not stat.S_ISREG(initial.st_mode):
            raise ManifestError(f"Input is not a regular file: {path}")
        digest = hashlib.sha256()
        with path.open("rb") as source:
            before = os.fstat(source.fileno())
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
            after = os.fstat(source.fileno())
        if not (_fingerprint(before) == _fingerprint(after)
                and _identity(initial) == _identity(before) == _identity(path.stat())):
            raise ManifestError(f"Input changed while hashing; retry: {path}")
    except OSError as exc:
        raise ManifestError(f"Cannot hash input {path}: {exc}") from exc
    return {"path": str(path), "size_bytes": after.st_size,
            "sha256": digest.hexdigest()}


def _git(repository, *arguments):
    try:
        result = subprocess.run(
            ["git", "--no-optional-locks", "-c", "core.fsmonitor=false",
             "-C", str(repository), *arguments],
            check=True, capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=10,
        )
    except FileNotFoundError as exc:
        raise ManifestError("Git is required to record the checkout revision.") from exc
    except subprocess.TimeoutExpired as exc:
        raise ManifestError("Git metadata collection exceeded 10 seconds.") from exc
    except subprocess.CalledProcessError as exc:
        raise ManifestError(f"Cannot inspect Git checkout: {exc.stderr.strip()}") from exc
    except OSError as exc:
        raise ManifestError(f"Cannot run Git: {exc}") from exc
    return result.stdout


def git_metadata(repository):
    root = _git(repository, "rev-parse", "--show-toplevel").strip()
    head = _git(repository, "rev-parse", "--verify", "HEAD").strip()
    dirty = bool(_git(repository, "status", "--porcelain=v1", "-z",
                      "--untracked-files=normal"))
    if head != _git(repository, "rev-parse", "--verify", "HEAD").strip():
        raise ManifestError("Git HEAD changed during collection; retry.")
    return {"repository": root, "head": head, "dirty": dirty,
            "dirty_scope": "tracked changes and non-ignored untracked files"}


def physical_memory_bytes():
    """Host physical memory, when POSIX sysconf exposes both required values."""
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
    except (AttributeError, ValueError, OSError):
        return None
    return pages * page_size if pages > 0 and page_size > 0 else None


def build_manifest(repository, config=None, artifacts=(), command=None, seed=None):
    # Validate every explicitly supplied file before producing output.
    config_record = hash_file(config) if config is not None else None
    artifact_records = [hash_file(path) for path in artifacts]
    packages = sorted(
        ({"name": dist.metadata.get("Name"), "version": dist.version}
         for dist in metadata.distributions()),
        key=lambda item: (item["name"] or "", item["version"] or ""),
    )
    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "provenance_scope": (
            "Checkout, environment, and files observed at manifest creation; "
            "not proof of the environment or command that produced an earlier run."
        ),
        "git": git_metadata(repository),
        "environment": {
            "python_version": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine() or None,
            "logical_cpu_count": os.cpu_count(),
            "host_physical_memory_bytes": physical_memory_bytes(),
            "installed_distributions": packages,
        },
        "inputs": {"config": config_record, "artifacts": artifact_records},
        "declared_run": {
            "command": command, "seed": seed,
            "verified_by_this_tool": False,
            "execution_performed_by_this_tool": False,
        },
    }


def write_manifest(manifest, output):
    """Publish complete JSON exclusively, using a same-directory hard link."""
    output = Path(output).absolute()
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=output.parent,
            prefix=f".{output.name}.", suffix=".tmp", delete=False,
        ) as destination:
            temporary = Path(destination.name)
            json.dump(manifest, destination, indent=2, ensure_ascii=False,
                      allow_nan=False)
            destination.write("\n")
            destination.flush()
            os.fsync(destination.fileno())
        # Unlike replace(), link() refuses even a concurrently created output.
        # Unsupported filesystems fail explicitly rather than weakening this.
        os.link(temporary, output)
    except FileExistsError as exc:
        raise ManifestError(f"Refusing to overwrite existing output: {output}") from exc
    except OSError as exc:
        raise ManifestError(
            f"Cannot publish manifest {output}: {exc}. The output directory must "
            "exist and support hard links."
        ) from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path,
                        help="New JSON sidecar path; never overwrites a file")
    parser.add_argument("--repo", type=Path, default=Path.cwd(),
                        help="Git checkout to inspect (default: current directory)")
    parser.add_argument("--config", type=Path, help="Existing configuration to hash")
    parser.add_argument("--artifact", type=Path, action="append", default=[],
                        help="Existing result/data artifact to hash; repeatable")
    parser.add_argument("--command", help="Declared run command, never executed")
    parser.add_argument("--seed", type=int, help="Declared run seed, not verified")
    args = parser.parse_args(argv)
    try:
        manifest = build_manifest(args.repo, args.config, args.artifact,
                                  args.command, args.seed)
        write_manifest(manifest, args.output)
    except ManifestError as exc:
        parser.exit(1, f"research_manifest: {exc}\n")
    print(f"Created {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
