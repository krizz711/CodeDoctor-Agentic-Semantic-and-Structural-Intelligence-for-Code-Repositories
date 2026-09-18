from __future__ import annotations

import json
from urllib import request

from app.models.catalog import ModelSpec


class ModelRequestError(RuntimeError):
    """Raised when a BYOK provider request cannot be completed."""


def chat_completion(model: ModelSpec, messages: list[dict[str, str]], *, timeout: int = 90,
                    temperature: float = 0.0, max_tokens: int = 2048) -> str:
    import os

    api_key = os.getenv(model.api_key_env)
    if not api_key and model.provider != "ollama":
        raise ModelRequestError(f"Missing API key environment variable: {model.api_key_env}")

    if model.provider == "anthropic":
        system_messages = [message["content"] for message in messages if message.get("role") == "system"]
        conversation = [message for message in messages if message.get("role") != "system"]
        body = {
            "model": model.model,
            "messages": conversation,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if system_messages:
            body["system"] = "\n\n".join(system_messages)
        headers = {
            "x-api-key": api_key or "",
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
            "User-Agent": "CodeDoctor/1.0",
        }
    else:
        body = {
            "model": model.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "CodeDoctor/1.0",
        }
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

    payload = json.dumps(body).encode("utf-8")
    request_data = request.Request(
        model.endpoint,
        data=payload,
        headers=headers,
        method="POST",
    )
    try:
        with request.urlopen(request_data, timeout=timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
    except Exception as error:
        raise ModelRequestError(f"Provider request failed for {model.alias}: {error}") from error

    try:
        if model.provider == "anthropic":
            return "".join(item["text"] for item in result["content"] if item.get("type") == "text")
        return result["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as error:
        raise ModelRequestError(f"Provider returned an unsupported response for {model.alias}") from error
