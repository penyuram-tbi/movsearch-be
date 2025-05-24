from elasticsearch import Elasticsearch
from app.core.config import settings

def get_elasticsearch_client() -> Elasticsearch:
    """Create and return an Elasticsearch client"""
    return Elasticsearch(
        settings.ELASTICSEARCH_URL,
        api_key=settings.ELASTICSEARCH_API_KEY
    )

# Create a singleton instance of the Elasticsearch client
es_client = get_elasticsearch_client()