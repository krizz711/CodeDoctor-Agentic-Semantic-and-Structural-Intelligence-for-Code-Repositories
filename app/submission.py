"""Shared judge-facing configuration. No model is loaded by this module."""
from pathlib import Path
import os
import yaml
from app.pipeline.execution_rerank import ExecutionConfig

SUBMISSION_CONFIG = Path(__file__).resolve().parents[1] / "config/submission.yaml"

def get_default_config_path() -> Path:
    return SUBMISSION_CONFIG

def read_config(path: Path) -> dict:
    """The YAML config at ``path``.

    A missing or malformed config used to become ``{}``, which silently turned
    execution verification off; both are now errors.
    """
    if not path.exists():
        raise FileNotFoundError(f"config not found: {path}")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (yaml.YAMLError, UnicodeDecodeError) as error:
        raise ValueError(f"config {path} is not valid YAML: {error}") from error
    # An empty file used to load as {} and silently changed what runs: no
    # execution pass, and the encoder fell back to a different (smaller) model.
    if not isinstance(data, dict) or not data:
        raise ValueError(f"config {path} is empty or not a YAML mapping")
    return data


def get_pipeline_config(config_path: Path | None = None) -> dict:
    """The ``pipeline`` block of the frozen config, with safe fallbacks."""
    data = read_config(config_path or get_default_config_path())
    block = data.get("pipeline", {}) or {}
    return {
        "execution_rerank": bool(block.get("execution_rerank", False)),
        "execution_candidates": int(block.get("execution_candidates", 10)),
        "execution_timeout_seconds": float(block.get("execution_timeout_seconds", 2.0)),
        "execution_numeric_tolerance": (
            None
            if block.get("execution_numeric_tolerance") in (None, "", False)
            else float(block["execution_numeric_tolerance"])
        ),
        "execution_max_dense_gap": (
            None
            if block.get("execution_max_dense_gap") in (None, "", False)
            else float(block["execution_max_dense_gap"])
        ),
        # 0 or absent means "one per CPU"; parallelism never changes a result.
        "execution_workers": int(block.get("execution_workers", 0) or 0),
        # v2 switches -- all default off, so config/frozen.yaml is unchanged.
        "execution_call_style": bool(block.get("execution_call_style", False)),
        "execution_extended_stdin": bool(block.get("execution_extended_stdin", False)),
        "execution_judge_rules": bool(block.get("execution_judge_rules", False)),
        "execution_timeout_mode": str(block.get("execution_timeout_mode", "wall")),
        "execution_deepen_to": int(block.get("execution_deepen_to", 0) or 0),
        "execution_skip_interactive": bool(block.get("execution_skip_interactive", False)),
        "execution_blank_line_fallback": bool(block.get("execution_blank_line_fallback", False)),
        "execution_judge_like_io": bool(block.get("execution_judge_like_io", False)),
        # PRE-5 hubness correction: off unless a bank is configured.
        "hubness_bank": block.get("hubness_bank") or None,
        "hubness_beta": float(block.get("hubness_beta", 0.3)),
        "hubness_k": int(block.get("hubness_k", 10)),
    }


def resolve_workers(requested: int) -> int:
    """``0`` means one worker per physical core.

    Sandbox runs are CPU-bound and their wall-clock timeouts are load-sensitive
    (experiments/RESULTS.md, "Timeouts are load-dependent"), so hyperthreads do
    not count.  Falls back to the logical count when psutil cannot tell.
    """
    if requested > 0:
        return requested
    try:
        import psutil

        physical = psutil.cpu_count(logical=False)
    except ImportError:
        physical = None
    return physical or os.cpu_count() or 1


def execution_config(path: Path | None = None) -> ExecutionConfig:
    values = get_pipeline_config(path)
    names = {"execution_rerank": "enabled", "execution_candidates": "max_candidates",
             "execution_timeout_seconds": "timeout_seconds"}
    kwargs = {names.get(k, k.removeprefix("execution_")): v
              for k, v in values.items() if k.startswith("execution_")}
    kwargs["workers"] = resolve_workers(kwargs["workers"])
    return ExecutionConfig(**kwargs)


def identify(tier: str | None = None, path: Path | None = None) -> dict:
    """Which configuration and model a response was produced under.

    Reads the file only (no model is loaded): the path, a short content hash,
    the tier, the model and revision that tier selects, and the accepted
    execution switches, so a saved result can be tied to a configuration.
    """
    import hashlib

    path = path or SUBMISSION_CONFIG
    data = read_config(path)
    tier = tier or os.environ.get("CODEDOCTOR_TIER", "accuracy")
    encoder = dict(data.get("encoder", {}) or {})
    if tier == "interactive":
        encoder.update(data.get("interactive_encoder", {}) or {})
    pipeline = get_pipeline_config(path)
    try:
        shown = str(path.resolve().relative_to(Path(__file__).resolve().parents[1])).replace("\\", "/")
    except ValueError:
        shown = str(path)
    return {
        "configuration": shown,
        "config_sha256": hashlib.sha256(path.read_bytes()).hexdigest()[:16],
        "tier": tier,
        "model": encoder.get("production_model"),
        "revision": encoder.get("revision"),
        "execution": {key.removeprefix("execution_"): value for key, value in pipeline.items()
                      if key.startswith("execution_")},
    }
