from pathlib import Path

from app.config import Settings


def test_settings_use_healthwise_data_defaults():
    settings = Settings()

    assert settings.data_dir == Path("/Users/sam/Documents/Ellipsis-Care/data")
    assert settings.index_path == Path(".ally_index/index.json")
    assert settings.chunk_size == 1200
    assert settings.chunk_overlap == 200


def test_settings_read_azure_openai_environment(monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "key")
    monkeypatch.setenv("AZURE_OPENAI_BASE_URL", "https://example.openai.azure.com/")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o-mini")
    monkeypatch.setenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME", "text-embedding-3-small")

    settings = Settings()

    assert settings.azure_openai_api_key == "key"
    assert str(settings.azure_openai_base_url) == "https://example.openai.azure.com/"
    assert settings.azure_openai_deployment_name == "gpt-4o-mini"
    assert settings.azure_openai_embedding_deployment_name == "text-embedding-3-small"

