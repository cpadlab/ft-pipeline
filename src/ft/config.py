from functools import lru_cache
from pydantic import AnyHttpUrl, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Runtime settings loaded from environment variables or a `.env` file.
    """

    AI_BASE_URL: AnyHttpUrl = Field(default="http://localhost:8000/v1")
    AI_API_KEY: SecretStr = Field(..., min_length=10)
    AI_MODEL: str = Field(default="default-model", min_length=1)
    AI_VERIFY_SSL: bool = Field(default=True)
    AI_MAX_CONCURRENCY: int = Field(default=4, ge=1, le=100)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    """
    Returns a cached singleton instance of `Settings`.
    """
    return Settings()