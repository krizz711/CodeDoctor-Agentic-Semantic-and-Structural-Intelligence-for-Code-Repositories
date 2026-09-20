from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ModelSpec:
    alias: str
    display_name: str
    provider: str
    model: str
    endpoint: str
    api_key_env: str
    notes: str = ""

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ModelSpec":
        return cls(
            alias=value["alias"],
            display_name=value["display_name"],
            provider=value["provider"],
            model=value["model"],
            endpoint=value["endpoint"],
            api_key_env=value["api_key_env"],
            notes=value.get("notes", ""),
        )

    def configured(self) -> bool:
        if self.provider == "ollama" and self.endpoint.startswith("http://localhost"):
            return True
        return bool(os.getenv(self.api_key_env))


def load_catalog(path: str | Path = "config/models.json") -> dict[str, ModelSpec]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return {item["alias"]: ModelSpec.from_dict(item) for item in payload["models"]}
