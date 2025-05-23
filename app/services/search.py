from elasticsearch import Elasticsearch
from typing import List, Dict, Any, Optional
from app.db.elasticsearch import es_client
from app.models.movie import Movie
from app.services.vector import get_embedding
from app.core.config import settings

import math
# from numpy import dot
# from numpy.linalg import norm

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
            if field == "genres" and isinstance(value, list):
                # Special handling for genres list - create AND LOGIC
                genre_terms = []
                for genre in value:
                    # Match genres that contain any of the requested genres
                    genre_terms.append({"match_phrase": {"genres": genre}})
                
                if genre_terms:
                    filter_clauses.append({
                        "bool": {
                            "must": genre_terms,
                            # "minimum_should_match": 1
                        }
                    })
            elif isinstance(value, list):
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
            "fields": ["title^3", "overview^2", "genres", "tagline", "director", "cast", "production_countries^2", "spoken_languages^2"],
            "fuzziness": "AUTO"
        }
    }]
    # print(settings.ELASTICSEARCH_URL)
    # print(settings.ELASTICSEARCH_API_KEY)
    # Add filters if provided
    filter_clauses = []
    if filters:
        for field, value in filters.items():
            if field == "genres" and isinstance(value, list):
                # Special handling for genres list - create AND LOGIC
                genre_terms = []
                for genre in value:
                    # Match genres that contain any of the requested genres
                    genre_terms.append({"match_phrase": {"genres": genre}})
                
                if genre_terms:
                    filter_clauses.append({
                        "bool": {
                            "must": genre_terms,
                            # "minimum_should_match": 1
                        }
                    })
            elif isinstance(value, list):
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

def hybrid_search(
    query: str,
    size: int = 10,
    bm25_multiplier: float = 0.5,
    vector_multiplier: float = 0.5,
    filters: Optional[Dict[str, Any]] = None,
    es: Elasticsearch = es_client
) -> List[Movie]:
    """
    Hybrid search combining keyword (BM25) and semantic (vector) search
    with re-ranking based on combined scores
    """
    # Step 1: Get embedding for semantic search
    vector = get_embedding(query)
    
    # Step 2: Define BM25 query (retrieve more results than needed for re-ranking)
    retrieve_size = min(size * 3, 100)  # Get more results to re-rank
    
    keyword_query = {
        "bool": {
            "must": [{
                "multi_match": {
                    "query": query,
                    "fields": ["title^3", "overview^2", "genres", "tagline", "director", "cast", "production_countries^2", "spoken_languages^2"],
                    "fuzziness": "AUTO"
                }
            }]
        }
    }
    
    # Apply filters if provided
    if filters:
        filter_clauses = []
        for field, value in filters.items():
            if field == "genres" and isinstance(value, list):
                # Special handling for genres list - create AND LOGIC
                genre_terms = []
                for genre in value:
                    # Match genres that contain any of the requested genres
                    genre_terms.append({"match_phrase": {"genres": genre}})
                
                if genre_terms:
                    filter_clauses.append({
                        "bool": {
                            "must": genre_terms
                            # "minimum_should_match": 1
                        }
                    })
            elif isinstance(value, list):
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
        
        keyword_query["bool"]["filter"] = filter_clauses
    
    # Step 3: Execute BM25 search to get initial results
    result = es.search(
        index=settings.INDEX_NAME,
        body={
            "size": retrieve_size,
            "query": keyword_query,
            "_source": True  # Include the complete document
        }
    )
    # print(result)
    # print("Ini adalah hasil bm25 esearch")
    
    # Step 4: Re-rank results using vector similarity
    hits = result["hits"]["hits"]
    reranked_results = []
    
    # Calculate max BM25 score for normalization
    max_bm25_score = max([hit["_score"] for hit in hits]) if hits else 1.0
    
    for hit in hits:
        source = hit["_source"]
        bm25_score = hit["_score"] / max_bm25_score  # Normalize BM25 score (0-1 range)
        
        # Get document embedding and calculate cosine similarity
        doc_embedding = source.get("embedding", [])
        if doc_embedding:
            # Calculate cosine similarity manually since we're outside ES query
            similarity = cosine_similarity_manual(vector, doc_embedding)
            
            # Combine scores with weights
            combined_score = (bm25_score * bm25_multiplier) + (similarity * vector_multiplier)
            
            reranked_results.append({
                "source": source,
                "bm25_score": bm25_score,
                "vector_score": similarity,
                "combined_score": combined_score
            })
    
    # Step 5: Sort by combined score and take top results
    reranked_results.sort(key=lambda x: x["combined_score"], reverse=True)
    top_results = reranked_results[:size]
    
    # Step 6: Convert to Movie objects
    movies = []
    for result in top_results:
        source = result["source"]
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
            score=result["combined_score"]
        ))
    
    return movies

def cosine_similarity_manual(vec1: list, vec2: list) -> float:
    """Calculate cosine similarity between two vectors"""
    if not vec1 or not vec2 or len(vec1) != len(vec2):
        return 0.0
    
    # Calculate dot product
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    
    # Calculate magnitudes
    magnitude1 = math.sqrt(sum(a * a for a in vec1))
    magnitude2 = math.sqrt(sum(b * b for b in vec2))
    
    # Avoid division by zero
    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0
    
    # Cosine similarity
    similarity = dot_product / (magnitude1 * magnitude2)
    
    # Normalize to 0-1 range (original is -1 to 1)
    return (similarity + 1) / 2

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