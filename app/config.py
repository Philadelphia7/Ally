from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    app_name: str = "Ally Healthwise RAG"
    data_dir: Path = Path("/Users/sam/Documents/Ellipsis-Care/data")
    index_path: Path = Path(".ally_index/index.json")
    chunk_size: int = 1200
    chunk_overlap: int = 200
    retrieval_top_k: int = 5

    azure_openai_api_key: str | None = Field(default=None, alias="AZURE_OPENAI_API_KEY")
    azure_openai_base_url: str | None = Field(default=None, alias="AZURE_OPENAI_BASE_URL")
    azure_openai_deployment_name: str | None = Field(
        default=None,
        alias="AZURE_OPENAI_DEPLOYMENT_NAME",
    )
    azure_openai_embedding_deployment_name: str = Field(
        default="text-embedding-3-small",
        alias="AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME",
    )
    azure_openai_api_version: str = Field(
        default="2024-10-21",
        alias="AZURE_OPENAI_API_VERSION",
    )

    document_intelligence_endpoint: str | None = Field(
        default=None,
        alias="DOCUMENT_INTELLIGENCE_ENDPOINT",
    )
    document_intelligence_subscription_key: str | None = Field(
        default=None,
        alias="DOCUMENT_INTELLIGENCE_SUBSCRIPTION_KEY",
    )

    speech_key: str | None = Field(default=None, alias="SPEECH_KEY")
    speech_region: str | None = Field(default=None, alias="SPEECH_REGION")
    speech_voice_name: str = Field(default="en-NG-EzinneNeural", alias="SPEECH_VOICE_NAME")

    medication_database_url: str | None = Field(default=None, alias="MEDICATION_DATABASE_URL")

    @property
    def azure_openai_configured(self) -> bool:
        return bool(
            self.azure_openai_api_key
            and self.azure_openai_base_url
            and self.azure_openai_deployment_name
        )

    @property
    def document_intelligence_configured(self) -> bool:
        return bool(
            self.document_intelligence_endpoint
            and self.document_intelligence_subscription_key
        )

    @property
    def speech_configured(self) -> bool:
        return bool(self.speech_key and self.speech_region)


@lru_cache
def get_settings() -> Settings:
    return Settings()

