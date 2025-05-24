import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from typing import List, Dict, Any
import os
import gc
import logging
import torch

logger = logging.getLogger(__name__)

# Set cache paths
os.environ['TRANSFORMERS_CACHE'] = '/app/model_cache'
os.environ['HF_HOME'] = '/app/model_cache'
# Lazy loading for model and tokenizer
_model = None
_tokenizer = None

def get_model_and_tokenizer():
    """GPU-optimized model loading with 4-bit quantization"""
    global _model, _tokenizer
    
    if _model is None or _tokenizer is None:
        model_name = "Qwen/Qwen2.5-0.5B-Instruct"
        
        logger.info("Loading GPU-optimized 4-bit quantized model...")
        
        # GPU-optimized quantization config
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        
        try:
            # Load tokenizer
            _tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir='/app/model_cache',
                trust_remote_code=True
            )
            
            # Load model with GPU optimization
            _model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map="auto",  # Auto GPU mapping
                torch_dtype=torch.float16,
                cache_dir='/app/model_cache',
                trust_remote_code=True
            )
            
            if _tokenizer.pad_token is None:
                _tokenizer.pad_token = _tokenizer.eos_token
            
            # Log GPU usage
            if torch.cuda.is_available():
                memory_used = torch.cuda.memory_allocated() / 1024**3
                logger.info(f"✅ Model loaded on GPU, using {memory_used:.1f}GB VRAM")
            else:
                logger.info("✅ Model loaded on CPU")
                
        except Exception as e:
            logger.error(f"❌ Error loading model: {e}")
            raise e
    
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
        "and interesting insights across the provided movies. Provide the summary directly without any introductory phrases. "
        "Do not include any conversational filler or introductory sentences. Get straight to the point."
    )
    
    user_prompt = (
        f"Here are some movies to summarize:\\n\\n{formatted_movies}\\n\\n"
        f"Please provide a concise, informative summary of these movies using Markdown formatting. "
        f"The entire summary, including all requested details, should be structured and complete, fitting naturally without being truncated. "
        f"The summary should:\\n"
        f"1. Give an overview of what type of movies are shown.\\n"
        f"2. Highlight any notable directors, themes, or patterns.\\n"
        f"3. Mention the top-rated film(s) in the results.\\n"
        f"4. Provide brief context on any common themes or genres.\\n"
        f"Summarize the collection as a whole, but you can include brief details for individual movies as appropriate to cover the points above.\\n\\n"
        f"Make the response related to the movies provided and query."
        f"Make the summary engaging and informative, similar to summary boxes in search results, using Markdown for structure (e.g., headings, lists, bolding)."
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
            max_new_tokens=250,  # Limit the summary length
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
    
    # GPU memory cleanup
    del model_inputs, generated_ids
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()

    logger.debug("GPU memory cleaned")
    
    # Clean up any trailing/leading whitespace
    return response.strip()