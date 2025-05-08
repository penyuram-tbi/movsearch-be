from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List

class QueryRequest(BaseModel):
    query: str
    size: Optional[int] = 10
    min_score: Optional[float] = 0.0
    filters: Optional[Dict[str, Any]] = None

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