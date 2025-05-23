from pydantic import BaseModel
from typing import List
from app.models.movie import Movie

class MovieSummaryRequest(BaseModel):
    movies: List[Movie]
    query: str = ""

class MovieSummaryResponse(BaseModel):
    summary: str
    query: str = ""
    movie_count: int