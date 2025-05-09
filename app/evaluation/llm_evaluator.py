import os
import torch
from typing import List, Dict, Any, Optional, Tuple
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer
from app.models.movie import Movie

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMEvaluator:
    """Uses QWen 3 0.5B to evaluate search relevance"""
    
    def __init__(self, model_name: str = "Qwen/Qwen2.5-0.5B"):
        """Initialize the LLM evaluator with QWen model"""
        logger.info(f"Initializing LLM evaluator with model: {model_name}")
        
        # Check if CUDA is available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, 
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )
            logger.info(f"Successfully loaded {model_name}")
            self.model_loaded = True
        except Exception as e:
            logger.error(f"Error loading QWen model: {e}")
            logger.warning("Will use simple heuristic relevance assessment instead.")
            self.model_loaded = False
    
    def evaluate_relevance(self, query: str, movie: Movie, query_intent: str = None) -> int:
        """
        Evaluate if a movie is relevant to a query using QWen LLM.
        Returns 1 (relevant) or 0 (not relevant)
        """
        if not self.model_loaded:
            return self._heuristic_evaluate(query, movie, query_intent)
        
        # Build the prompt for QWen
        prompt = self._build_prompt(query, movie, query_intent)
        
        try:
            # Generate response
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=5,
                    temperature=0.1,
                    num_return_sequences=1
                )
            
            response = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            
            # Extract binary relevance from response
            binary_score = self._extract_binary_score(response)
            return binary_score
        
        except Exception as e:
            logger.error(f"Error using QWen for relevance evaluation: {e}")
            return self._heuristic_evaluate(query, movie, query_intent)
    
    def _build_prompt(self, query: str, movie: Movie, query_intent: str = None) -> str:
        """Build prompt for QWen to evaluate relevance"""
        
        intent_explanation = ""
        if query_intent:
            intent_explanation = f"The user's search intent appears to be focused on {query_intent}."
        
        return f"""You are an expert movie recommendation system. Evaluate if the following movie is relevant to the user's search query.

Query: "{query}"
{intent_explanation}

Movie Information:
- Title: {movie.title}
- Year: {movie.release_date.split('-')[0] if movie.release_date else 'Unknown'}
- Director: {movie.director}
- Cast: {movie.cast[:200]}...
- Genre: {movie.genres}
- Plot: {movie.overview[:300]}...

Is this movie relevant to the search query? Respond with 'Yes' if it's relevant, or 'No' if it's not relevant.
Consider both exact matches and semantic relevance. A movie is relevant if it directly matches the search terms OR addresses the underlying information need of the query.

Answer (Yes/No): """
    
    def _extract_binary_score(self, response: str) -> int:
        """Extract binary score (0 or 1) from model response"""
        response = response.strip().lower()
        
        # If response clearly contains yes/no, use that
        if "yes" in response and "no" not in response:
            return 1
        elif "no" in response and "yes" not in response:
            return 0
        
        # Check for other affirmative/negative words
        positive_words = ["relevant", "appropriate", "suitable", "match", "related", "applicable"]
        negative_words = ["irrelevant", "inappropriate", "unsuitable", "unrelated"]
        
        for word in positive_words:
            if word in response:
                return 1
                
        for word in negative_words:
            if word in response:
                return 0
        
        # Default to 0 (not relevant) if unclear
        return 0
    
    def _heuristic_evaluate(self, query: str, movie: Movie, query_intent: str = None) -> int:
        """Fallback heuristic relevance assessment when LLM is unavailable"""
        query = query.lower()
        query_words = set(query.split())
        
        # Remove stopwords
        stopwords = {"the", "a", "an", "and", "in", "on", "at", "with", "about", "for", "to", "of"}
        query_words = {word for word in query_words if word not in stopwords}
        
        if not query_words:
            return 0  # If no significant words in query
        
        # Check title first - high value match
        if any(word in movie.title.lower() for word in query_words):
            return 1
            
        # Check director
        if query_intent == "director" and any(word in movie.director.lower() for word in query_words):
            return 1
            
        # Check cast
        if query_intent == "actor" and any(word in movie.cast.lower() for word in query_words):
            return 1
            
        # Check genre
        if query_intent == "genre" and any(word in movie.genres.lower() for word in query_words):
            return 1
        
        # Check overview
        if movie.overview and any(word in movie.overview.lower() for word in query_words):
            return 1
        
        # More complex matching for non-explicit matches
        # E.g., "war movies" should match "World War II drama"
        if query_intent == "theme" or query_intent == "plot":
            overview_wordset = set(movie.overview.lower().split()) if movie.overview else set()
            genre_wordset = set(movie.genres.lower().split()) if movie.genres else set()
            combined_wordset = overview_wordset.union(genre_wordset)
            
            # Special cases
            if "war" in query.lower() and any(war_term in combined_wordset for war_term in ["war", "battle", "combat", "military", "army", "soldier"]):
                return 1
                
            if "sci-fi" in query.lower() and any(sci_term in combined_wordset for sci_term in ["space", "alien", "future", "robot", "technology"]):
                return 1
        
        return 0  # Default to not relevant