llm_evaluation_queries = [
    # Specific film titles
    {"query": "The Shawshank Redemption", "intent": "title"},
    {"query": "Inception", "intent": "title"},
    {"query": "The Godfather", "intent": "title"},
    {"query": "Interstellar", "intent": "title"},
    {"query": "Jurassic Park", "intent": "title"},
    
    # Partial/fuzzy titles
    {"query": "Shawshank", "intent": "title"},
    {"query": "Lord of Rings", "intent": "title"},
    {"query": "Star war", "intent": "title"},
    {"query": "Mission Imposible", "intent": "title"},
    {"query": "Harry Poter", "intent": "title"},
    
    # Director searches
    {"query": "Christopher Nolan movies", "intent": "director"},
    {"query": "films directed by Quentin Tarantino", "intent": "director"},
    {"query": "Steven Spielberg war movies", "intent": "director"},
    {"query": "Martin Scorsese gangster films", "intent": "director"},
    {"query": "David Fincher psychological thrillers", "intent": "director"},
    
    # Actor searches
    {"query": "movies with Tom Hanks", "intent": "actor"},
    {"query": "Leonardo DiCaprio best films", "intent": "actor"},
    {"query": "Meryl Streep drama", "intent": "actor"},
    {"query": "action movies with Keanu Reeves", "intent": "actor"},
    {"query": "Samuel L Jackson films", "intent": "actor"},
    
    # Genre-based queries
    {"query": "best science fiction movies", "intent": "genre"},
    {"query": "horror films about haunted houses", "intent": "genre"},
    {"query": "romantic comedies with wedding", "intent": "genre"},
    {"query": "animated movies for children", "intent": "genre"},
    {"query": "crime thrillers with plot twists", "intent": "genre"},
    
    # Thematic searches
    {"query": "movies about artificial intelligence", "intent": "theme"},
    {"query": "films about time travel", "intent": "theme"},
    {"query": "movies with twist ending", "intent": "theme"},
    {"query": "films about mental illness", "intent": "theme"},
    {"query": "heist movies", "intent": "theme"},
    
    # Time period searches
    {"query": "best movies from the 90s", "intent": "time_period"},
    {"query": "classic 80s comedies", "intent": "time_period"},
    {"query": "2000s superhero films", "intent": "time_period"},
    {"query": "1970s crime movies", "intent": "time_period"},
    {"query": "movies from 2022", "intent": "time_period"},
    
    # Mixed queries
    {"query": "action movies with explosions and car chases", "intent": "plot"},
    {"query": "drama films with strong female leads", "intent": "plot"},
    {"query": "funny comedies with college students", "intent": "plot"},
    {"query": "suspenseful movies with unexpected endings", "intent": "plot"},
    {"query": "historical films about ancient rome", "intent": "plot"},
    
    # Location-based searches
    {"query": "movies set in New York", "intent": "location"},
    {"query": "films in Paris", "intent": "location"},
    {"query": "movies filmed in Japan", "intent": "location"},
    {"query": "italian movies", "intent": "location"},
    {"query": "indonesian horror films", "intent": "location"},
    
    # Rating-based searches
    {"query": "highest rated action movies", "intent": "rating"},
    {"query": "best romantic films of all time", "intent": "rating"},
    {"query": "top sci-fi movies with great special effects", "intent": "rating"},
    {"query": "critically acclaimed documentaries", "intent": "rating"},
    {"query": "underrated comedy movies", "intent": "rating"}
]

# Smaller set for weight optimization (fewer queries for faster execution)
llm_weight_optimization_queries = [
    {"query": "The Shawshank Redemption", "intent": "title"},
    {"query": "Christopher Nolan movies", "intent": "director"},
    {"query": "movies with Tom Hanks", "intent": "actor"},
    {"query": "best science fiction movies", "intent": "genre"},
    {"query": "movies about artificial intelligence", "intent": "theme"},
    {"query": "action movies with explosions and car chases", "intent": "plot"},
    {"query": "movies set in New York", "intent": "location"},
    {"query": "highest rated action movies", "intent": "rating"}
]