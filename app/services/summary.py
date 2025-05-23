import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from typing import List, Dict, Any

# Lazy loading for model and tokenizer
_model = None
_tokenizer = None

def get_model_and_tokenizer():
    """Initialize and return the model and tokenizer with lazy loading"""
    global _model, _tokenizer
    
    if _model is None or _tokenizer is None:
        model_name = "Qwen/Qwen2.5-0.5B-Instruct"
        
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
    
        _model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto"
        )
        
        _tokenizer = AutoTokenizer.from_pretrained(model_name)

        if _tokenizer.pad_token is None:
            _tokenizer.pad_token = _tokenizer.eos_token
    
    return _model, _tokenizer

def format_movie_data(movies: List[Dict[Any, Any]]) -> str:
    """Format movie data into a readable text for the LLM prompt"""
    formatted_text = ""
    
    for i, movie in enumerate(movies[:5]):  # Limiting to top 5 movies to avoid token limits
        title = movie.get("title", "Unknown Title")
        release_date = movie.get("release_date", "Unknown Year")
        year = release_date.split("-")[0] if release_date and "-" in release_date else "Unknown Year"
        overview = movie.get("overview", "No overview available")
        genres = movie.get("genres", "Unknown Genres")
        vote_average = movie.get("vote_average", "N/A")
        director = movie.get("director", "Unknown Director")
        
        formatted_text += f"Movie {i+1}:\n"
        formatted_text += f"Title: {title} ({year})\n"
        formatted_text += f"Directed by: {director}\n"
        formatted_text += f"Genres: {genres}\n"
        formatted_text += f"Rating: {vote_average}/10\n"
        formatted_text += f"Overview: {overview}\n\n"
    
    return formatted_text

def create_movie_summary(movies: List[Dict[Any, Any]], query: str = "") -> str:
    """
    Generate a summary of the given movies based on the search query using Qwen LLM
    
    Args:
        movies: List of movie data dictionaries
        query: The original search query (optional)
        
    Returns:
        A summary of the movies
    """
    if not movies:
        return "No movies provided for summarization."
    
    # Get the model and tokenizer
    model, tokenizer = get_model_and_tokenizer()
    
    # Format movie data for the prompt
    formatted_movies = format_movie_data(movies)
    
    # Create the prompt for the LLM
    system_prompt = (
        "You are a helpful movie recommendation assistant specialized in providing concise, informative, "
        "and engaging summaries of movies. Your summaries should highlight patterns, notable films, "
        "and interesting insights across the provided movies."
    )
    
    user_prompt = (
        f"Here are some movies to summarize:\n\n{formatted_movies}\n\n"
        f"Please provide a concise, informative summary of these movies (about 25 words). "
        f"The summary should:\n"
        f"1. Give an overview of what type of movies are shown if asked\n"
        f"2. Highlight any notable directors, themes, or patterns, if asked\n"
        f"3. Mention the top-rated film(s) in the results if asked\n"
        f"4. Provide brief context on any common themes or genres\n"
        f"DONT JUST FOCUS ON ONE MOVIE, but rather summarize the collection as a whole.\n"
        f"Make the response related to the movies provided and query.\n"
        f"Make the summary engaging and informative, like the summary boxes that appear in Google search results."
    )
    
    # Add query context if provided
    if query:
        user_prompt = f"I searched for movies related to: \"{query}\"\n\n" + user_prompt
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    # Apply chat template
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # Tokenize and generate
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    with torch.no_grad():  # No need to track gradients for inference
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=120,  # Limit the summary length
            temperature=0.7,     # Add some creativity but keep it factual
            top_p=0.9,           # Use nucleus sampling for more natural text
            repetition_penalty=1.2  # Avoid repetition
        )
    
    # Extract only the new tokens (the generated response)
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    
    # Decode the response
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    # Clean up any trailing/leading whitespace
    return response.strip()