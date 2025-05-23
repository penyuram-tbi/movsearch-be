from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    API_V1_STR: str
    PROJECT_NAME: str
    
    # Elasticsearch configuration
    ELASTICSEARCH_URL: str
    ELASTICSEARCH_API_KEY: str
    INDEX_NAME: str
    
    # Vector model configuration
    VECTOR_MODEL_NAME: str
    VECTOR_DIMENSIONS: int
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True
        env_prefix = ""
        extra = "ignore"

settings = Settings()