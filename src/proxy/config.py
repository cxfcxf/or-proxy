from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(".env", str(Path.home() / ".sieg.env")), extra="ignore"
    )

    openrouter_api_key: str
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    ranker_model: str = "anthropic/claude-sonnet-4.6"
    preferred_model: str | None = None
    poll_interval_seconds: int = 86400
    host: str = "127.0.0.1"
    port: int = 8787
    http_referer: str = "https://github.com/cxfcxf/or-proxy"
    x_title: str = "or-proxy"


settings = Settings()
