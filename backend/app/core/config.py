"""
Configuration settings for RAGFlow implementation.
"""
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with trio-specific configurations."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False
    )
    
    # Application
    PROJECT_NAME: str = "RAGFlow Implementation"
    API_V1_STR: str = "/api/v1"
    DEBUG: bool = False
    
    # Database
    DATABASE_URL: str = Field(
        default="postgresql+psycopg2://cognify:secret@postgres:5432/cognify_db",
        description="PostgreSQL database URL"
    )
    
    # Infinity DB (Vector Database)
    INFINITY_HOST: str = "infinity"
    INFINITY_PORT: int = 23817
    INFINITY_DATABASE: str = "default"
    
    # Elasticsearch (Sparse Search)
    ELASTICSEARCH_HOST: str = "elasticsearch"
    ELASTICSEARCH_PORT: int = 9200
    ELASTICSEARCH_INDEX: str = "ragflow_documents"
    
    # Redis (Caching)
    REDIS_HOST: str = "redis"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_PASSWORD: Optional[str] = None
    
    # Yandex Cloud ML
    YANDEX_API_KEY: Optional[str] = None
    YANDEX_FOLDER_ID: Optional[str] = None
    YANDEX_LLM_MODEL: str = "yandexgpt"
    YANDEX_EMBEDDING_MODEL: str = "text-search-doc"
    
    # Trio-specific settings
    MAX_CONCURRENT_TASKS: int = Field(default=5, description="Max concurrent document processing tasks")
    MAX_CONCURRENT_CHUNK_BUILDERS: int = Field(default=1, description="Max concurrent chunk builders")
    MAX_CONCURRENT_MINIO: int = Field(default=10, description="Max concurrent MinIO operations")
    MAX_CONCURRENT_CHATS: int = Field(default=10, description="Max concurrent LLM chat requests")
    
    # ONNX Models Configuration
    ONNX_MODELS_PATH: str = "/app/models"
    ONNX_MODELS_REPO: str = "InfiniFlow/deepdoc"
    DEEPDOC_ENABLED: bool = True
    DEEPDOC_CONFIDENCE_THRESHOLD: float = 0.7
    
    # Document Processing
    DEFAULT_CHUNK_SIZE: int = 1000
    DEFAULT_CHUNK_OVERLAP: int = 200
    MAX_DOCUMENT_SIZE_MB: int = 100
    SUPPORTED_FORMATS: list[str] = Field(
        default_factory=lambda: [
            "pdf", "docx", "xlsx", "pptx", "txt", "md", "html", "json"
        ]
    )
    
    # Search Configuration
    HYBRID_SEARCH_ENABLED: bool = True
    SPARSE_WEIGHT: float = 0.3
    DENSE_WEIGHT: float = 0.7
    RERANKING_ENABLED: bool = True
    MAX_SEARCH_RESULTS: int = 20
    MAX_RERANK_RESULTS: int = 5
    
    # RAPTOR Configuration
    RAPTOR_ENABLED: bool = False
    RAPTOR_MAX_LEVELS: int = 3
    RAPTOR_CLUSTER_THRESHOLD: float = 0.8
    
    # Knowledge Graph
    KG_ENABLED: bool = True
    KG_DATABASE_PATH: str = "/app/data/knowledge_graph"
    KG_ENTITY_CONFIDENCE_THRESHOLD: float = 0.8
    KG_RELATION_CONFIDENCE_THRESHOLD: float = 0.7
    
    # Caching
    LLM_CACHE_TTL: int = 3600  # 1 hour
    EMBEDDING_CACHE_TTL: int = 86400  # 24 hours
    SEARCH_CACHE_TTL: int = 1800  # 30 minutes
    
    # Performance
    ENABLE_PERFORMANCE_MONITORING: bool = True
    LOG_LEVEL: str = "INFO"
    
    # File Storage
    STORAGE_PATH: str = "/app/storage"
    TEMP_PATH: str = "/app/tmp"
    
    # Language Processing
    SPACY_MODEL_RU: str = "ru_core_news_md"
    SPACY_MODEL_EN: str = "en_core_web_md"
    DEFAULT_LANGUAGE: str = "ru"
    
    # Rate Limiting
    RATE_LIMIT_REQUESTS_PER_MINUTE: int = 60
    RATE_LIMIT_BURST: int = 10


settings = Settings()