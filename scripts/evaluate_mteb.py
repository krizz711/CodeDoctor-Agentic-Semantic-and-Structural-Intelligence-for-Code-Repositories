from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any
import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.retrieval.encode import CodeEncoder, FROZEN_CONFIG_PATH, DEFAULT_CONFIG_PATH, _select_device


def get_default_config_path() -> Path:
    if FROZEN_CONFIG_PATH.exists():
        return FROZEN_CONFIG_PATH
    return DEFAULT_CONFIG_PATH


def get_default_model(config_path: Path | None = None) -> str:
    path = config_path or get_default_config_path()
    if path.exists():
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
            return data.get("encoder", {}).get("production_model", "codefuse-ai/F2LLM-v2-0.6B")
        except Exception:
            pass
    return "codefuse-ai/F2LLM-v2-0.6B"


def create_mteb_model(code_encoder: CodeEncoder):
    import mteb
    from mteb.models.abs_encoder import AbsEncoder
    from mteb.types import Array, BatchedInput, PromptType
    from torch.utils.data import DataLoader

    try:
        model_meta = mteb.get_model_meta(code_encoder.model_name)
    except Exception:
        model_meta = None

    if model_meta is None:
        model_meta = type("SimpleModelMeta", (), {
            "name": code_encoder.model_name,
            "revision": "main",
            "similarity_fn_name": "cosine",
        })()

    class CodeDoctorMTEBEncoder(AbsEncoder):
        def __init__(self, encoder: CodeEncoder) -> None:
            self.encoder = encoder
            self.model = encoder.model
            self.mteb_model_meta = model_meta

        def encode(
            self,
            inputs: DataLoader[BatchedInput],
            *,
            task_metadata: Any = None,
            hf_split: str = "test",
            hf_subset: str = "default",
            prompt_type: PromptType | None = None,
            **kwargs: Any,
        ) -> Array:
            is_query = (
                prompt_type == PromptType.query
                or str(prompt_type) == "query"
                or str(prompt_type) == "PromptType.query"
            )
            mode = "query" if is_query else "document"

            embeddings_list: list[np.ndarray] = []
            for batch in inputs:
                if isinstance(batch, dict):
                    texts = batch.get("text", batch.get("sentences", []))
                elif isinstance(batch, (list, tuple)):
                    texts = list(batch)
                else:
                    texts = [str(batch)]

                cleaned = [t if isinstance(t, str) else str(t) for t in texts]
                batch_emb = self.encoder.encode(cleaned, mode=mode, batch_size=len(cleaned))
                embeddings_list.append(batch_emb)

            if not embeddings_list:
                return np.empty((0, 1024), dtype=np.float32)

            return np.vstack(embeddings_list)

        def similarity(self, embeddings1: Any, embeddings2: Any) -> Any:
            arr1 = np.asarray(embeddings1, dtype=np.float32)
            arr2 = np.asarray(embeddings2, dtype=np.float32)
            return self.encoder.similarity(arr1, arr2)

    return CodeDoctorMTEBEncoder(code_encoder)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate official MTEB AppsRetrieval result JSON using CodeEncoder.")
    parser.add_argument("--model", default=None, help="Model override (defaults to production_model from frozen config)")
    parser.add_argument("--config", type=Path, default=None, help="Path to config YAML")
    parser.add_argument("--output", type=Path, default=Path("appsretrieval_results.json"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto", help="Compute device (auto/cuda/cpu)")
    args = parser.parse_args()

    try:
        import mteb
    except ImportError as error:
        raise SystemExit(
            "Install evaluator dependencies first: pip install -r requirements-eval.txt"
        ) from error

    config_path = args.config or get_default_config_path()
    model_name = args.model or get_default_model(config_path)
    device = _select_device(args.device)

    print(f"Loading CodeEncoder with model: {model_name} on device: {device} (config: {config_path})")
    encoder = CodeEncoder(model_name=model_name, device=device, config_path=config_path)
    mteb_model = create_mteb_model(encoder)

    print("Fetching task AppsRetrieval...")
    task = mteb.get_task("AppsRetrieval")
    print("Evaluating with MTEB...")
    results = mteb.evaluate(mteb_model, [task], encode_kwargs={"batch_size": args.batch_size})
    task_result = list(results.task_results)[0]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(task_result.to_dict(), default=str, indent=2), encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
