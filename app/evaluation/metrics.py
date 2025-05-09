from typing import List, Dict, Any, Optional, Tuple
import json
import numpy as np
import os
import logging
from datetime import datetime
from app.models.movie import Movie
from app.services.search import semantic_search, keyword_search, hybrid_search
from app.evaluation.llm_evaluator import LLMEvaluator

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize LLM evaluator
evaluator = LLMEvaluator()

def calculate_ndcg(relevance_scores: List[int], k: int = None) -> float:
    """
    Calculate Normalized Discounted Cumulative Gain (NDCG)
    
    Args:
        relevance_scores: List of binary relevance scores (0 or 1)
        k: Number of results to consider (default: len(relevance_scores))
    
    Returns:
        NDCG score between 0 and 1
    """
    if not relevance_scores:
        return 0.0
    
    if k is None:
        k = len(relevance_scores)
    else:
        k = min(k, len(relevance_scores))
    
    # Get relevance scores up to position k
    relevance_scores = relevance_scores[:k]
    
    # Calculate DCG
    dcg = sum((2**r - 1) / np.log2(i + 2) for i, r in enumerate(relevance_scores))
    
    # Calculate Ideal DCG (IDCG)
    # Sort relevance scores in descending order for ideal ranking
    ideal_relevance = sorted(relevance_scores, reverse=True)
    idcg = sum((2**r - 1) / np.log2(i + 2) for i, r in enumerate(ideal_relevance))
    
    # Return NDCG
    return dcg / idcg if idcg > 0 else 0.0

def evaluate_search_results(query: str, query_intent: str, movies: List[Movie], top_k: int = 10) -> Dict[str, Any]:
    """
    Evaluate search results using LLM for relevance judgments
    
    Args:
        query: Search query
        query_intent: Intent of the query (e.g., "title", "director")
        movies: List of movies returned by search
        top_k: Number of top results to evaluate
    
    Returns:
        Dictionary with evaluation metrics
    """
    if not movies:
        return {
            "precision@k": 0.0,
            "ndcg": 0.0,
            "relevant_count": 0,
            "total_results": 0,
            "query": query,
            "query_intent": query_intent
        }
    
    # Take top k results
    top_results = movies[:min(top_k, len(movies))]
    
    # Get relevance judgments from LLM
    relevance_scores = []
    for movie in top_results:
        # Get binary relevance score (0 or a)
        relevance = evaluator.evaluate_relevance(query, movie, query_intent)
        relevance_scores.append(relevance)
    
    # Calculate metrics
    relevant_count = sum(relevance_scores)
    precision_at_k = relevant_count / len(top_results) if top_results else 0
    ndcg = calculate_ndcg(relevance_scores)
    
    return {
        "precision@k": precision_at_k,
        "ndcg": ndcg,
        "relevant_count": relevant_count,
        "total_results": len(top_results),
        "relevance_scores": relevance_scores,
        "search_scores": [movie.score for movie in top_results],
        "query": query,
        "query_intent": query_intent
    }

def compare_search_methods(
    query_data: List[Dict[str, str]], 
    top_k: int = 10,
    results_dir: str = None
) -> Dict[str, Any]:
    """
    Compare different search methods using LLM-based evaluation
    
    Args:
        query_data: List of dictionaries with query and intent
        top_k: Number of top results to evaluate
        results_dir: Directory to save results
    
    Returns:
        Dictionary with comparison results
    """
    results = {
        "semantic": [],
        "keyword": [],
        "hybrid": [],
        "queries": [q["query"] for q in query_data],
        "summary": {}
    }
    
    # Create results directory
    if results_dir is None:
        script_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        results_dir = os.path.join(script_dir, 'evaluation_results')
    
    os.makedirs(results_dir, exist_ok=True)
    
    # Set up logging to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(results_dir, f'llm_evaluation_{timestamp}.log')
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    
    logger.info(f"Starting LLM-based evaluation with {len(query_data)} queries")
    logger.info(f"Using QWEN model: {evaluator.model_loaded}")
    
    for query_item in query_data:
        query = query_item["query"]
        intent = query_item.get("intent", "general")
        
        logger.info(f"Evaluating query: {query} (Intent: {intent})")
        
        try:
            # Run each search method
            semantic_results = semantic_search(query=query, size=top_k)
            keyword_results = keyword_search(query=query, size=top_k)
            hybrid_results = hybrid_search(
                query=query, 
                size=top_k,
                bm25_multiplier=0.5,
                vector_multiplier=0.5
            )
            
            # Log result titles for debugging
            logger.info(f"Semantic search results: {[m.title for m in semantic_results[:3]]}")
            logger.info(f"Keyword search results: {[m.title for m in keyword_results[:3]]}")
            logger.info(f"Hybrid search results: {[m.title for m in hybrid_results[:3]]}")
            
            # Evaluate each method
            semantic_eval = evaluate_search_results(query, intent, semantic_results, top_k)
            keyword_eval = evaluate_search_results(query, intent, keyword_results, top_k)
            hybrid_eval = evaluate_search_results(query, intent, hybrid_results, top_k)
            
            # Log evaluation results
            logger.info(f"Semantic - P@{top_k}: {semantic_eval['precision@k']:.2f}, NDCG: {semantic_eval['ndcg']:.2f}, Relevant: {semantic_eval['relevant_count']}/{semantic_eval['total_results']}")
            logger.info(f"Keyword - P@{top_k}: {keyword_eval['precision@k']:.2f}, NDCG: {keyword_eval['ndcg']:.2f}, Relevant: {keyword_eval['relevant_count']}/{keyword_eval['total_results']}")
            logger.info(f"Hybrid - P@{top_k}: {hybrid_eval['precision@k']:.2f}, NDCG: {hybrid_eval['ndcg']:.2f}, Relevant: {hybrid_eval['relevant_count']}/{hybrid_eval['total_results']}")
            
            # Store results
            results["semantic"].append(semantic_eval)
            results["keyword"].append(keyword_eval)
            results["hybrid"].append(hybrid_eval)
            
        except Exception as e:
            logger.error(f"Error evaluating query '{query}': {e}")
    
    # Calculate summary statistics
    for method in ["semantic", "keyword", "hybrid"]:
        if not results[method]:  # Skip if no results for method
            continue
            
        method_precision = [r["precision@k"] for r in results[method]]
        method_ndcg = [r["ndcg"] for r in results[method]]
        
        results["summary"][method] = {
            "avg_precision": float(np.mean(method_precision)),
            "avg_ndcg": float(np.mean(method_ndcg)),
            "num_queries": len(query_data),
            "best_query": max(results[method], key=lambda x: x["precision@k"])["query"],
            "worst_query": min(results[method], key=lambda x: x["precision@k"])["query"]
        }
    
    # Save detailed results to JSON file
    json_file = os.path.join(results_dir, f'llm_detailed_results_{timestamp}.json')
    with open(json_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Evaluation completed. Results saved to {json_file}")
    
    return results

def find_optimal_weights(
    query_data: List[Dict[str, str]],
    weight_steps: int = 5,
    top_k: int = 10,
    results_dir: str = None
) -> Dict[str, Any]:
    """
    Find optimal weights for hybrid search using LLM-based evaluation
    
    Args:
        query_data: List of dictionaries with query and intent
        weight_steps: Number of steps for weight grid search
        top_k: Number of top results to evaluate
        results_dir: Directory to save results
    
    Returns:
        Dictionary with optimization results
    """
    # Create results directory
    if results_dir is None:
        script_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        results_dir = os.path.join(script_dir, 'evaluation_results')
    
    os.makedirs(results_dir, exist_ok=True)
    
    # Set up logging to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(results_dir, f'llm_weight_optimization_{timestamp}.log')
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    
    logger.info(f"Finding optimal weights with {len(query_data)} queries")
    
    best_avg_precision = 0
    best_avg_ndcg = 0
    best_weights_precision = (0.5, 0.5)
    best_weights_ndcg = (0.5, 0.5)
    all_results = {}
    
    # Test different weight combinations
    for i in range(weight_steps + 1):
        bm25_weight = i / weight_steps
        vector_weight = 1.0 - bm25_weight
        
        logger.info(f"Testing weights: BM25={bm25_weight:.2f}, Vector={vector_weight:.2f}")
        
        results_for_weight = []
        
        for query_item in query_data:
            query = query_item["query"]
            intent = query_item.get("intent", "general")
            
            try:
                hybrid_results = hybrid_search(
                    query=query,
                    size=top_k,
                    bm25_multiplier=bm25_weight,
                    vector_multiplier=vector_weight
                )
                
                eval_result = evaluate_search_results(query, intent, hybrid_results, top_k)
                results_for_weight.append(eval_result)
                
                logger.info(f"  Query: {query[:30]}... - P@{top_k}: {eval_result['precision@k']:.2f}, NDCG: {eval_result['ndcg']:.2f}")
                
            except Exception as e:
                logger.error(f"Error optimizing for query '{query}': {e}")
        
        if not results_for_weight:  # Skip if no results
            continue
            
        avg_precision = float(np.mean([r["precision@k"] for r in results_for_weight]))
        avg_ndcg = float(np.mean([r["ndcg"] for r in results_for_weight]))
        
        weight_key = f"bm25_{bm25_weight:.2f}_vector_{vector_weight:.2f}"
        all_results[weight_key] = {
            "avg_precision": avg_precision,
            "avg_ndcg": avg_ndcg,
            "weights": {"bm25": bm25_weight, "vector": vector_weight}
        }
        
        logger.info(f"Average precision: {avg_precision:.4f}, Average NDCG: {avg_ndcg:.4f}")
        
        if avg_precision > best_avg_precision:
            best_avg_precision = avg_precision
            best_weights_precision = (bm25_weight, vector_weight)
        
        if avg_ndcg > best_avg_ndcg:
            best_avg_ndcg = avg_ndcg
            best_weights_ndcg = (bm25_weight, vector_weight)
    
    # Save results to JSON file
    results = {
        "best_weights_precision": {
            "bm25": best_weights_precision[0],
            "vector": best_weights_precision[1]
        },
        "best_precision": best_avg_precision,
        "best_weights_ndcg": {
            "bm25": best_weights_ndcg[0],
            "vector": best_weights_ndcg[1]
        },
        "best_ndcg": best_avg_ndcg,
        "all_results": all_results,
        "queries": [q["query"] for q in query_data]
    }
    
    json_file = os.path.join(results_dir, f'llm_weight_optimization_{timestamp}.json')
    with open(json_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Weight optimization completed. Results saved to {json_file}")
    logger.info(f"Best weights (precision): BM25={best_weights_precision[0]:.2f}, Vector={best_weights_precision[1]:.2f} with precision {best_avg_precision:.4f}")
    logger.info(f"Best weights (NDCG): BM25={best_weights_ndcg[0]:.2f}, Vector={best_weights_ndcg[1]:.2f} with NDCG {best_avg_ndcg:.4f}")
    
    return results