from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.models.catalog import load_catalog


def main() -> None:
    parser = argparse.ArgumentParser(description="List CodeDoctor BYOK model aliases.")
    parser.add_argument("--configured-only", action="store_true")
    parser.add_argument("--catalog", type=Path, default=Path("config/models.json"))
    args = parser.parse_args()
    models = list(load_catalog(args.catalog).values())
    if args.configured_only:
        models = [model for model in models if model.configured()]
    for model in models:
        state = "configured" if model.configured() else "not configured"
        print(f"{model.alias}\t{model.display_name}\t{model.provider}\t{state}")
    print(f"Total: {len(models)}")


if __name__ == "__main__":
    main()
