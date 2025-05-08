from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Movie Search API"
    
    # Elasticsearch configuration
    ELASTICSEARCH_URL: str = "https://my-elasticsearch-project-dab849.es.us-central1.gcp.elastic.cloud:443"
    ELASTICSEARCH_API_KEY: str = "R0RWZnI1WUJfcGdtMkxLNGd0dGU6OXBtR3kyNncyZkVNQjFjcXAwODd5UQ=="
    INDEX_NAME: str = "semantic_documents"
    
    # Vector model configuration
    VECTOR_MODEL_NAME: str = "all-MiniLM-L6-v2"
    VECTOR_DIMENSIONS: int = 384

settings = Settings()