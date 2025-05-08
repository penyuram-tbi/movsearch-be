from elasticsearch import Elasticsearch
from typing import List, Dict, Any, Optional
from app.db.elasticsearch import es_client
from app.models.movie import Movie
from app.services.vector import get_embedding
from app.core.config import settings

def semantic_search(
    query: str,
    size: int = 10,
    min_score: float = 0.0,
    filters: Optional[Dict[str, Any]] = None,
    es: Elasticsearch = es_client
) -> List[Movie]:
    """
    Search for movies using semantic similarity
    """
    # Generate embedding for the query
    vector = get_embedding(query)
    
    # Base query with vector similarity
    script_query = {
        "script_score": {
            "query": {"match_all": {}},
            "script": {
                "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                "params": {"query_vector": vector}
            }
        }
    }
    
    # Apply filters if provided
    if filters:
        filter_clauses = []
        for field, value in filters.items():
            if isinstance(value, list):
                filter_clauses.append({"terms": {field: value}})
            elif isinstance(value, dict) and ("min" in value or "max" in value):
                range_filter = {"range": {field: {}}}
                if "min" in value:
                    range_filter["range"][field]["gte"] = value["min"]
                if "max" in value:
                    range_filter["range"][field]["lte"] = value["max"]
                filter_clauses.append(range_filter)
            else:
                filter_clauses.append({"term": {field: value}})
        
        # Use filtered query
        query_body = {
            "bool": {
                "must": script_query,
                "filter": filter_clauses
            }
        }
    else:
        query_body = script_query
    
    # Execute search
    result = es.search(
        index=settings.INDEX_NAME,
        body={
            "size": size,
            "query": query_body,
            "min_score": min_score
        }
    )
    
    # Process results
    movies = []
    for hit in result["hits"]["hits"]:
        source = hit["_source"]
        movies.append(Movie(
            id=str(source.get("id", "")),
            title=source.get("title", ""),
            overview=source.get("overview", ""),
            release_date=source.get("release_date", ""),
            vote_average=source.get("vote_average", 0),
            popularity=source.get("popularity", 0),
            genres=source.get("genres", ""),
            director=source.get("director", ""),
            cast=source.get("cast", ""),
            poster_path=source.get("poster_path", ""),
            tagline=source.get("tagline", ""),
            runtime=source.get("runtime", 0),
            imdb_rating=source.get("imdb_rating", 0),
            score=hit["_score"]
        ))
    
    return movies

def keyword_search(
    query: str,
    size: int = 10,
    filters: Optional[Dict[str, Any]] = None,
    es: Elasticsearch = es_client
) -> List[Movie]:
    """
    Search for movies using keyword matching
    """
    # Create the base query
    must_clauses = [{
        "multi_match": {
            "query": query,
            "fields": ["title^3", "overview^2", "genres", "tagline", "director", "cast"],
            "fuzziness": "AUTO"
        }
    }]
    
    # Add filters if provided
    filter_clauses = []
    if filters:
        for field, value in filters.items():
            if isinstance(value, list):
                filter_clauses.append({"terms": {field: value}})
            elif isinstance(value, dict) and ("min" in value or "max" in value):
                range_filter = {"range": {field: {}}}
                if "min" in value:
                    range_filter["range"][field]["gte"] = value["min"]
                if "max" in value:
                    range_filter["range"][field]["lte"] = value["max"]
                filter_clauses.append(range_filter)
            else:
                filter_clauses.append({"term": {field: value}})
    
    # Combine query parts
    query_body = {
        "bool": {
            "must": must_clauses
        }
    }
    
    if filter_clauses:
        query_body["bool"]["filter"] = filter_clauses
    
    # Execute search
    result = es.search(
        index=settings.INDEX_NAME,
        body={
            "size": size,
            "query": query_body
        }
    )
    
    # Process results
    movies = []
    for hit in result["hits"]["hits"]:
        source = hit["_source"]
        movies.append(Movie(
            id=str(source.get("id", "")),
            title=source.get("title", ""),
            overview=source.get("overview", ""),
            release_date=source.get("release_date", ""),
            vote_average=source.get("vote_average", 0),
            popularity=source.get("popularity", 0),
            genres=source.get("genres", ""),
            director=source.get("director", ""),
            cast=source.get("cast", ""),
            poster_path=source.get("poster_path", ""),
            tagline=source.get("tagline", ""),
            runtime=source.get("runtime", 0),
            imdb_rating=source.get("imdb_rating", 0),
            score=hit["_score"]
        ))
    
    return movies

def get_movie_by_id(movie_id: str, es: Elasticsearch = es_client) -> Optional[Movie]:
    """
    Get a movie by its ID
    """
    try:
        result = es.get(index=settings.INDEX_NAME, id=movie_id)
        source = result["_source"]
        # print(source)
        
        return Movie(
            id=str(source.get("id", "")),
            title=source.get("title", ""),
            overview=source.get("overview", ""),
            release_date=source.get("release_date", ""),
            vote_average=source.get("vote_average", 0),
            popularity=source.get("popularity", 0),
            genres=source.get("genres", ""),
            director=source.get("director", ""),
            cast=source.get("cast", ""),
            poster_path=source.get("poster_path", ""),
            tagline=source.get("tagline", ""),
            runtime=source.get("runtime", 0),
            imdb_rating=source.get("imdb_rating", 0),
            score=1.0
        )
    except:
        return None