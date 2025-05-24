from fastapi import APIRouter, HTTPException, Query, Depends
from typing import List
from app.models.movie import Movie, QueryRequest, KeywordSearchRequest
from app.services.search import semantic_search, keyword_search, get_movie_by_id, hybrid_search

router = APIRouter()

@router.post("/hybrid-search", response_model=List[Movie])
async def search_movies_hybrid(req: QueryRequest):
    """
    Search for movies using hybrid approach combining BM25 and vector similarity.
    Provides better results by leveraging both keyword matching and semantic understanding.
    """
    try:
        # Default weight is 50-50 but can be customized
        weights = getattr(req, "weights", {}) or {}  # Handle None case
        bm25_weight = weights.get("bm25", 0.5)
        vector_weight = weights.get("vector", 0.5)
        
        results = hybrid_search(
            query=req.query,
            size=req.size if hasattr(req, "size") else 10,
            bm25_multiplier=bm25_weight,
            vector_multiplier=vector_weight,
            filters=req.filters if hasattr(req, "filters") else None
        )
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Hybrid search failed: {str(e)}")
    
@router.post("/semantic-search", response_model=List[Movie])
async def search_movies_semantic(req: QueryRequest):
    """
    Search for movies using semantic similarity with the query.
    Supports filtering and advanced query options.
    """
    try:
        results = semantic_search(
            query=req.query,
            size=req.size,
            min_score=req.min_score,
            filters=req.filters
        )
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@router.post("/keyword-search", response_model=List[Movie])
async def search_movies_keyword(req: KeywordSearchRequest):
    """
    Search for movies using keyword matching in title, overview, and other fields.
    Supports basic filtering options via query parameters.
    """
    try:
        # Build filters dict from query parameters
        filters = {}
        if req.year_min is not None or req.year_max is not None:
            filters["year"] = {}
            if req.year_min is not None:
                filters["year"]["min"] = req.year_min
            if req.year_max is not None:
                filters["year"]["max"] = req.year_max
                
        if req.rating_min is not None:
            filters["vote_average"] = {"min": req.rating_min}
            
        if req.genres is not None:
            genres_list = [genre.strip() for genre in req.genres.split(',')]
            filters["genres"] = genres_list
            
        results = keyword_search(
            query=req.query,
            size=req.size,
            filters=filters if filters else None
        )
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@router.get("/{movie_id}", response_model=Movie)
async def get_movie(movie_id: str):
    """
    Get a specific movie by its ID
    """
    movie = get_movie_by_id(movie_id)
    if not movie:
        raise HTTPException(status_code=404, detail=f"Movie with ID {movie_id} not found")
    return movie