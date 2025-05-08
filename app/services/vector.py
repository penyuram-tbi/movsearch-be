from sentence_transformers import SentenceTransformer
from app.core.config import settings

# Initialize the model
model = SentenceTransformer(settings.VECTOR_MODEL_NAME)

def get_embedding(text: str) -> list:
    """Generate embedding vector for a text"""
    return model.encode(text).tolist()

def create_semantic_text(doc: dict) -> str:
    """Create a rich semantic text representation from a document"""
    semantic_text = f"{doc.get('title', '')} "
    semantic_text += f"{doc.get('overview', '')} "
    semantic_text += f"{doc.get('tagline', '')} "
    semantic_text += f"{doc.get('genres', '')} "
    semantic_text += f"Released on {doc.get('release_date', '')} "
    semantic_text += f"Released in {doc.get('year', '')} "
    semantic_text += f"Directed by {doc.get('director', '')} "
    
    # Add cast (limit to first 5 actors)
    cast = doc.get('cast', '')
    if cast:
        cast_list = cast.split(', ')[:5]
        semantic_text += f"Starring {', '.join(cast_list)}"
        
    return semantic_text.strip()