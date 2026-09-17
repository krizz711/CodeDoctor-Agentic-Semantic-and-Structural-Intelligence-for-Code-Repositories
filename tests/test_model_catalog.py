from app.models.catalog import load_catalog


def test_catalog_contains_requested_model_families():
    catalog = load_catalog()
    assert len(catalog) >= 30
    names = {model.display_name for model in catalog.values()}
    for expected in ("DeepSeek V4", "Kimi K2.6", "GLM-5.1", "Qwen3.7", "MiMo V2.5", "MiniMax M2.7", "GPT-5.5", "Gemini 3.5", "Grok"):
        assert expected in names


def test_catalog_does_not_require_secrets():
    catalog = load_catalog()
    assert all(model.api_key_env and model.endpoint and model.model for model in catalog.values())


def test_local_ollama_models_are_available_without_remote_keys():
    catalog = load_catalog()
    assert catalog["phi-4"].configured()
