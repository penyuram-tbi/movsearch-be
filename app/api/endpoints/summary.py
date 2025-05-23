from fastapi import APIRouter, Body

from app.models.summary import MovieSummaryRequest, MovieSummaryResponse
from app.services.summary import create_movie_summary

router = APIRouter()

@router.post("/summarize", response_model=MovieSummaryResponse)
async def summarize_movies(request: MovieSummaryRequest = Body(...)):
    """
    Generate a summary of the provided movies
    
    The frontend should send a list of movies and an optional query
    """
    # Extract movie data from request
    movies = [movie.dict() for movie in request.movies]
    query = request.query
    
    # Generate summary using the LLM
    summary = create_movie_summary(movies, query)
    
    # Return the summary along with metadata
    return MovieSummaryResponse(
        summary=summary,
        query=query,
        movie_count=len(movies)
    )