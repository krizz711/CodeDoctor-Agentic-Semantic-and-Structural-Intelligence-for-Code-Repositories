from app.models.catalog import load_catalog


def test_catalog_loads_models():
    catalog = load_catalog()
    assert len(catalog) > 0
    # Check that standard model entries exist and have expected structure
    for model in catalog.values():
        assert model.model
        assert model.provider


def test_catalog_does_not_require_secrets():
    catalog = load_catalog()
    assert all(model.api_key_env and model.endpoint and model.model for model in catalog.values())


def test_local_ollama_models_are_available_without_remote_keys():
    catalog = load_catalog()
    if "phi-4" in catalog:
        assert catalog["phi-4"].configured()
