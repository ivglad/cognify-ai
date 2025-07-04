from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding='utf-8')

    PROJECT_NAME: str = "Cognify AI"
    API_V1_STR: str = "/api/v1"

    # Database settings
    DATABASE_URL: str

    # Yandex Cloud settings
    YANDEX_API_KEY: str | None = None
    YANDEX_FOLDER_ID: str | None = None

    # GigaChat settings
    GIGACHAT_CREDENTIALS: str | None = None

    # OpenRouter settings
    OPENROUTER_API_KEY: str | None = None
    OPENROUTER_MODEL_NAME: str = "openai/gpt-4-turbo"

    # LLM Provider settings
    LLM_PROVIDER: str = "yandex"  # 'yandex', 'giga', 'openrouter'
    EMBEDDING_PROVIDER: str = "yandex" # 'yandex'

    # Feature Toggles
    ENABLE_QUERY_REWRITING: bool = True

    # Infinity DB settings
    INFINITY_HOST: str = "infinity"
    INFINITY_PORT: int = 23817

settings = Settings() 