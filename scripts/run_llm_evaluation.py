import os
import sys
import json
import time
from datetime import datetime

# Add the project root to the path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.insert(0, project_root)

from app.evaluation.metrics import compare_search_methods, find_optimal_weights
from app.evaluation.queries import llm_evaluation_queries, llm_weight_optimization_queries

def create_evaluation_summary(comparison_results, weight_results=None):
    """Create a human-readable summary of LLM-based evaluation results"""
    summary = []
    
    # Add basic header
    summary.append("=" * 80)
    summary.append("MOVIE SEARCH LLM EVALUATION SUMMARY")
    summary.append("=" * 80)
    summary.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    summary.append(f"Number of test queries: {len(comparison_results['queries'])}")
    summary.append("-" * 80)
    
    # Compare methods
    summary.append("\nSEARCH METHOD COMPARISON (LLM Binary Relevance):")
    summary.append("-" * 40)
    
    methods = ["semantic", "keyword", "hybrid"]
    metrics = [
        ("Average Precision@10", "avg_precision"),
        ("Average NDCG", "avg_ndcg")
    ]
    
    # Add a table header
    header = "Method".ljust(15)
    for metric_name, _ in metrics:
        header += f" | {metric_name}".ljust(25)
    summary.append(header)
    summary.append("-" * 90)
    
    # Add each method's results
    for method in methods:
        line = method.capitalize().ljust(15)
        for _, metric_key in metrics:
            value = comparison_results["summary"][method][metric_key]
            line += f" | {value:.4f}".ljust(25)
        summary.append(line)
    
    # Add best/worst query for each method
    summary.append("\nBEST/WORST QUERIES BY METHOD:")
    summary.append("-" * 40)
    
    for method in methods:
        summary.append(f"\n{method.capitalize()}:")
        summary.append(f"  Best query: \"{comparison_results['summary'][method]['best_query']}\"")
        summary.append(f"  Worst query: \"{comparison_results['summary'][method]['worst_query']}\"")
    
    # Add weight optimization results if available
    if weight_results:
        summary.append("\n\nWEIGHT OPTIMIZATION RESULTS:")
        summary.append("-" * 40)
        summary.append(f"Best weights for Precision@10:")
        summary.append(f"  BM25: {weight_results['best_weights_precision']['bm25']:.2f}")
        summary.append(f"  Vector: {weight_results['best_weights_precision']['vector']:.2f}")
        summary.append(f"  Resulting precision: {weight_results['best_precision']:.4f}")
        
        summary.append(f"\nBest weights for NDCG:")
        summary.append(f"  BM25: {weight_results['best_weights_ndcg']['bm25']:.2f}")
        summary.append(f"  Vector: {weight_results['best_weights_ndcg']['vector']:.2f}")
        summary.append(f"  Resulting NDCG: {weight_results['best_ndcg']:.4f}")
    
    # Add footer
    summary.append("\n" + "=" * 80)
    summary.append("END OF EVALUATION SUMMARY")
    summary.append("=" * 80)
    
    return "\n".join(summary)

def main():
    print("Starting LLM-based evaluation of search methods")
    
    # Create results directory
    results_dir = os.path.join(project_root, 'evaluation_results')
    os.makedirs(results_dir, exist_ok=True)
    
    # Run method comparison first with a subset of queries
    query_subset = llm_evaluation_queries[:10]  # Start with 10 queries for quick testing
    
    print(f"Running method comparison with {len(query_subset)} queries...")
    start_time = time.time()
    comparison_results = compare_search_methods(query_subset, top_k=5, results_dir=results_dir)
    comparison_time = time.time() - start_time
    print(f"Method comparison completed in {comparison_time:.2f} seconds")
    
    # Ask if user wants to run with more queries
    use_more_queries = input("Run with all queries? This will take longer. (y/n, default: n): ").lower() == 'y'
    
    if use_more_queries:
        print(f"Running with all {len(llm_evaluation_queries)} queries...")
        start_time = time.time()
        comparison_results = compare_search_methods(llm_evaluation_queries, top_k=5, results_dir=results_dir)
        comparison_time = time.time() - start_time
        print(f"Full evaluation completed in {comparison_time:.2f} seconds")
    
    # Optionally run weight optimization
    run_weight_optimization = input("Run weight optimization? (y/n, default: n): ").lower() == 'y'
    weight_results = None
    
    if run_weight_optimization:
        print(f"Running weight optimization with {len(llm_weight_optimization_queries)} queries...")
        start_time = time.time()
        weight_results = find_optimal_weights(
            llm_weight_optimization_queries, 
            weight_steps=5,
            top_k=5,
            results_dir=results_dir
        )
        optimization_time = time.time() - start_time
        print(f"Weight optimization completed in {optimization_time:.2f} seconds")
    
    # Create and save human-readable summary
    summary = create_evaluation_summary(comparison_results, weight_results)
    summary_file = os.path.join(results_dir, f'llm_evaluation_summary_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')
    
    with open(summary_file, 'w') as f:
        f.write(summary)
    
    print(f"\nResults saved to {results_dir}")
    print(f"Summary saved to {summary_file}")
    
    # Print the summary to console
    print("\n" + summary)

if __name__ == "__main__":
    main()