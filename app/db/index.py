from elasticsearch import Elasticsearch
from app.core.config import settings
from app.db.elasticsearch import es_client

def create_index(es: Elasticsearch = es_client) -> None:
    """Create Elasticsearch index with mappings if it doesn't exist"""
    try:
        es.indices.create(
            index=settings.INDEX_NAME,
            body={
                "mappings": {
                    "properties": {
                        "id": {"type": "keyword"},
                        "title": {"type": "text", "analyzer": "english"},
                        "vote_average": {"type": "float"},
                        "vote_count": {"type": "float"},
                        "status": {"type": "keyword"},
                        "release_date": {"type": "date", "format": "yyyy-MM-dd||yyyy"},
                        "revenue": {"type": "long"},
                        "runtime": {"type": "integer"},
                        "budget": {"type": "long"},
                        "imdb_id": {"type": "keyword"},
                        "original_language": {"type": "keyword"},
                        "original_title": {"type": "text"},
                        "overview": {"type": "text", "analyzer": "english"},
                        "popularity": {"type": "float"},
                        "tagline": {"type": "text", "analyzer": "english"},
                        "genres": {"type": "text", "analyzer": "english"},
                        "production_companies": {"type": "text"},
                        "production_countries": {"type": "text"},
                        "spoken_languages": {"type": "text"},
                        "cast": {"type": "text"},
                        "director": {"type": "text"},
                        "director_of_photography": {"type": "text"},
                        "writers": {"type": "text"},
                        "producers": {"type": "text"},
                        "music_composer": {"type": "text"},
                        "imdb_rating": {"type": "float"},
                        "imdb_votes": {"type": "float"},
                        "poster_path": {"type": "keyword"},
                        "year": {"type": "integer"},
                        "profit": {"type": "float"},
                        "roi": {"type": "float"},
                        "imdb_url": {"type": "keyword"},
                        "embedding": {"type": "dense_vector", "dims": settings.VECTOR_DIMENSIONS}
                    }
                }
            },
            ignore=400  # Ignore error if index already exists
        )
        print(f"Created index: {settings.INDEX_NAME}")
        return True
    except Exception as e:
        print(f"Error creating index: {e}")
        return False

def delete_index(es: Elasticsearch = es_client) -> bool:
    """Delete the Elasticsearch index"""
    try:
        es.indices.delete(index=settings.INDEX_NAME, ignore=[404])
        print(f"Deleted index: {settings.INDEX_NAME}")
        return True
    except Exception as e:
        print(f"Error deleting index: {e}")
        return False