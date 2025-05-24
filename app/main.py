from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.endpoints import movies, summary
from app.core.config import settings
import os
import logging
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
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
@app.on_event("startup")
async def startup_event():
    try:
        os.environ['TRANSFORMERS_CACHE'] = '/app/model_cache'
        os.environ['HF_HOME'] = '/app/model_cache'
        
        from app.services.summary import get_model_and_tokenizer
        get_model_and_tokenizer()
        logger.info("Models loaded successfully")
        
    except Exception as e:
        logger.error(f"Startup error: {e}")

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

@app.get("/health")
async def health_check():
    try:
        import torch
        gpu_available = torch.cuda.is_available()
        
        from app.services.summary import _model, _tokenizer
        models_loaded = _model is not None and _tokenizer is not None
        
        return {
            "status": "healthy",
            "gpu_available": gpu_available,
            "models_loaded": models_loaded
        }
    except Exception as e:
        return {"status": "error", "detail": str(e)}