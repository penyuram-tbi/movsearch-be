from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List

class QueryRequest(BaseModel):
    query: str
    size: Optional[int] = 10
    min_score: Optional[float] = 0.0
    filters: Optional[Dict[str, Any]] = None
    weights: Optional[Dict[str, float]] = None

    class Config:
        schema_extra = {
            "example": {
                "query": "science fiction with aliens",
                "size": 10,
                "filters": {
                    "year": {"min": 2010, "max": 2023},
                    "vote_average": {"min": 7.0},
                    "genres": ["Science Fiction", "Action"]
                },
                "weights": {"bm25": 0.7, "vector": 0.3}
            }
        }

class KeywordSearchRequest(BaseModel):
    query: str
    size: int = 10
    year_min: Optional[int] = None
    year_max: Optional[int] = None
    rating_min: Optional[float] = None
    genres: Optional[str] = None
    
    class Config:
        schema_extra = {
            "example": {
                "query": "action adventure",
                "size": 10,
                "year_min": 2000,
                "year_max": 2022,
                "rating_min": 7.5,
                "genres": "Action, Adventure"
            }
        }
        
class Movie(BaseModel):
    id: str
    title: str
    overview: Optional[str] = None
    release_date: Optional[str] = None
    vote_average: Optional[float] = None
    popularity: Optional[float] = None
    genres: Optional[str] = None
    director: Optional[str] = None
    cast: Optional[str] = None
    poster_path: Optional[str] = None
    tagline: Optional[str] = None
    runtime: Optional[int] = None
    imdb_rating: Optional[float] = None
    score: float = 0.0

class MovieList(BaseModel):
    movies: List[Movie]
    total: int
    query: Optional[str] = None