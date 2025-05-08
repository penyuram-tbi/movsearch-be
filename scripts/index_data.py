import pandas as pd
import time
import sys
import os

# Add the project root to the path so we can import app modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.db.elasticsearch import es_client
from app.db.index import create_index, delete_index
from app.services.vector import create_semantic_text, get_embedding
from app.core.config import settings

def index_data(csv_path: str, recreate_index: bool = False):
    """Index data from CSV into Elasticsearch"""
    
    # Recreate index if requested
    if recreate_index:
        delete_index()
    
    # Create index if it doesn't exist
    create_index()
    
    # Load data from CSV
    try:
        df = pd.read_csv(csv_path)
        documents = df.to_dict(orient="records")
        print(f"Loaded {len(documents)} documents from {csv_path}")
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return False
    
    # Index all documents
    indexed_count = 0
    for i, doc in enumerate(documents):
        try:
            # Create semantic text for embedding
            semantic_text = create_semantic_text(doc)
            
            # Generate embedding vector
            vector = get_embedding(semantic_text)
            
            # Create document for indexing with all fields and handle missing values
            movie_doc = {k: ('' if pd.isna(v) else v) for k, v in doc.items()}
            
            # Add the embedding vector
            movie_doc["embedding"] = vector
            
            # Handle numeric field conversion
            for field in ["vote_average", "vote_count", "revenue", "runtime", "budget", 
                        "popularity", "imdb_rating", "imdb_votes", "year", "profit", "roi"]:
                if field in movie_doc and movie_doc[field] != '':
                    try:
                        movie_doc[field] = float(movie_doc[field])
                    except (ValueError, TypeError):
                        movie_doc[field] = 0.0
            
            # Index the document
            es_client.index(index=settings.INDEX_NAME, id=doc["id"], body=movie_doc)
            indexed_count += 1
            
            # Print progress 
            print(f"Indexed {i + 1}/{len(documents)}: {doc.get('title', '')}")
            
            # Small pause to avoid overwhelming ES
            time.sleep(0.1)
            
        except Exception as e:
            print(f"Error indexing document {doc.get('id', i)}: {e}")
    
    print(f"Successfully indexed {indexed_count}/{len(documents)} documents")
    return True

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Index movie data into Elasticsearch")
    parser.add_argument("--csv", default="app/data/testSample.csv", help="Path to CSV file")
    parser.add_argument("--recreate", action="store_true", help="Recreate the index (delete existing)")
    
    args = parser.parse_args()
    
    index_data(args.csv, args.recreate)