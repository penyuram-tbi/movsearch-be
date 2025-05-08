from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Movie Search API"
    
    # Elasticsearch configuration
    ELASTICSEARCH_URL: str
    ELASTICSEARCH_API_KEY: str
    INDEX_NAME: str = "semantic_documents"
    
    # Vector model configuration
    VECTOR_MODEL_NAME: str = "all-MiniLM-L6-v2"
    VECTOR_DIMENSIONS: int = 384
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True
        env_prefix = ""
        extra = "ignore"

settings = Settings()