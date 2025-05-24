from sentence_transformers import SentenceTransformer
from app.core.config import settings

# Initialize the model
model = SentenceTransformer(settings.VECTOR_MODEL_NAME)

def get_embedding(text: str) -> list:
    """Generate embedding vector for a text"""
    return model.encode(text).tolist()

def create_semantic_text(doc: dict) -> str:
    """Create a rich semantic text representation from a document using all attributes"""
    
    semantic_text = ""
    
    # Judul dan identitas film
    if doc.get('title', ''):
        semantic_text += f"Title: {doc.get('title', '')}. "
    if doc.get('original_title', '') and doc.get('original_title', '') != doc.get('title', ''):
        semantic_text += f"Original title: {doc.get('original_title', '')}. "
    
    # Deskripsi dan sinopsis
    if doc.get('overview', ''):
        semantic_text += f"Description: {doc.get('overview', '')}. "
    if doc.get('tagline', ''):
        semantic_text += f"Tagline: {doc.get('tagline', '')}. "
    
    # Crew utama
    if doc.get('director', ''):
        semantic_text += f"Directed by {doc.get('director', '')}. "
    if doc.get('writers', ''):
        semantic_text += f"Written by {doc.get('writers', '')}. "
    if doc.get('producers', ''):
        semantic_text += f"Produced by {doc.get('producers', '')}. "
    if doc.get('music_composer', ''):
        semantic_text += f"Music by {doc.get('music_composer', '')}. "
    if doc.get('director_of_photography', ''):
        semantic_text += f"Cinematography by {doc.get('director_of_photography', '')}. "
    
    # Cast - menggunakan seluruh info cast jika tersedia
    cast = doc.get('cast', '')
    if cast:
        cast_list = cast.split(', ')
        if len(cast_list) > 15:  # Jika cast sangat panjang
            # Ambil 15 aktor pertama
            main_cast = cast_list[:15]
            semantic_text += f"Starring {', '.join(main_cast)}. "
            semantic_text += f"The film also features {len(cast_list) - 15} other actors. "
        else:
            semantic_text += f"Starring {cast}. "
    
    # Genre dan kategori
    if doc.get('genres', ''):
        semantic_text += f"Genres: {doc.get('genres', '')}. "
    
    # Info produksi
    if doc.get('production_companies', ''):
        semantic_text += f"Produced by {doc.get('production_companies', '')}. "
    if doc.get('production_countries', ''):
        semantic_text += f"Produced in {doc.get('production_countries', '')}. "
    if doc.get('spoken_languages', ''):
        semantic_text += f"Languages: {doc.get('spoken_languages', '')}. "
    
    # Info rilis
    if doc.get('release_date', ''):
        semantic_text += f"Released on {doc.get('release_date', '')}. "
    if doc.get('year', ''):
        semantic_text += f"Released in {doc.get('year', '')}. "
    if doc.get('status', ''):
        semantic_text += f"Status: {doc.get('status', '')}. "
    
    # Metrics dan statistik
    if doc.get('runtime', 0):
        hours = doc.get('runtime', 0) // 60
        minutes = doc.get('runtime', 0) % 60
        if hours > 0:
            semantic_text += f"Duration: {hours} hour{'s' if hours > 1 else ''}"
            if minutes > 0:
                semantic_text += f" and {minutes} minute{'s' if minutes > 1 else ''}"
        else:
            semantic_text += f"Duration: {minutes} minute{'s' if minutes > 1 else ''}"
        semantic_text += ". "
    
    # Rating dan popularitas
    vote_average = doc.get('vote_average', 0)
    vote_count = doc.get('vote_count', 0)
    if vote_average and vote_count:
        semantic_text += f"Rated {vote_average}/10 from {int(vote_count)} votes. "
    
    imdb_rating = doc.get('imdb_rating', 0)
    imdb_votes = doc.get('imdb_votes', 0)
    if imdb_rating and imdb_votes:
        semantic_text += f"IMDb rating: {imdb_rating} from {int(imdb_votes)} votes. "
    
    popularity = doc.get('popularity', 0)
    if popularity:
        semantic_text += f"Popularity score: {popularity}. "
    
    # Finansial
    budget = doc.get('budget', 0)
    if budget and budget > 0:
        budget_millions = budget / 1000000
        semantic_text += f"Budget: ${budget_millions:.1f} million. "
    
    revenue = doc.get('revenue', 0)
    if revenue and revenue > 0:
        revenue_millions = revenue / 1000000
        semantic_text += f"Box office: ${revenue_millions:.1f} million. "
    
    profit = doc.get('profit', 0)
    if profit:
        if profit > 0:
            profit_millions = profit / 1000000
            semantic_text += f"Made a profit of ${profit_millions:.1f} million. "
        elif profit < 0:
            loss_millions = abs(profit) / 1000000
            semantic_text += f"Made a loss of ${loss_millions:.1f} million. "
    
    roi = doc.get('roi', 0)
    if roi and roi > 0:
        semantic_text += f"Return on investment: {roi:.1f}%. "
    
    # Link dan referensi
    if doc.get('imdb_url', ''):
        semantic_text += f"IMDb: {doc.get('imdb_url', '')}. "
    if doc.get('imdb_id', ''):
        semantic_text += f"IMDb ID: {doc.get('imdb_id', '')}. "
    
    return semantic_text.strip()