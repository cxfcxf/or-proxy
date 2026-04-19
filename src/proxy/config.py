from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    openrouter_api_key: str
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    poll_interval_seconds: int = 86400
    host: str = "127.0.0.1"
    port: int = 8787
    http_referer: str = "http://localhost"
    x_title: str = "hermes-free-proxy"


settings = Settings()
