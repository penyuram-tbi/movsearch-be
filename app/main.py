from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.endpoints import movies, summary
from app.core.config import settings

# Create FastAPI app
app = FastAPI(
    title=settings.PROJECT_NAME,
    description="API for semantic and keyword search of movies",
    version="2.0.0",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Include routers
app.include_router(
    movies.router,
    prefix=f"{settings.API_V1_STR}/movies",
    tags=["movies"]
)
app.include_router(
    summary.router,
    prefix=f"{settings.API_V1_STR}/search",
    tags=["summarize"]
)

@app.get("/", tags=["status"])
async def root():
    """
    Root endpoint to check if the API is running
    """
    return {
        "status": "ok",
        "message": f"{settings.PROJECT_NAME} is running",
        "version": "1.0.0"
    }