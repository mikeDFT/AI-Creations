"""
Data collection and processing for GA experiments.

This module handles the collection, processing, and organization
of experimental results for statistical analysis.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any
import json
from pathlib import Path


class DataCollector:
    """Collects and processes experimental data."""
    
    def create_results_dataframe(self, results: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Create a DataFrame from experiment results.
        
        Args:
            results: List of experiment result dictionaries
            
        Returns:
            DataFrame with processed results
        """
        processed_results = []
        
        for result in results:
            # Extract basic metrics
            processed_result = {
                'config_name': result['config_name'],
                'function_name': result['function_name'],
                'representation': result['representation'],
                'crossover_type': result['crossover_type'],
                'run_number': result['run_number'],
                'seed': result['seed'],
                'best_fitness': result['best_fitness'],
                'final_generation': result['final_generation'],
                'fitness_evaluations': result['fitness_evaluations'],
                'execution_time': result['execution_time']
            }
            
            # Calculate additional metrics
            convergence_history = result['convergence_history']
            if convergence_history:
                processed_result['initial_fitness'] = convergence_history[0]
                processed_result['improvement'] = convergence_history[0] - result['best_fitness']
                processed_result['convergence_rate'] = self.calculate_convergence_rate(convergence_history)
                processed_result['success'] = result['best_fitness'] < 1e-6  # Near-optimal threshold
            
            processed_results.append(processed_result)
        
        return pd.DataFrame(processed_results)
    
    def calculate_convergence_rate(self, fitness_history: List[float]) -> float:
        """
        Calculate convergence rate as improvement per generation.
        
        Args:
            fitness_history: List of best fitness values per generation
            
        Returns:
            Convergence rate
        """
        if len(fitness_history) < 2:
            return 0.0
        
        initial_fitness = fitness_history[0]
        final_fitness = fitness_history[-1]
        generations = len(fitness_history)
        
        if initial_fitness == 0:
            return 0.0
        
        return (initial_fitness - final_fitness) / (initial_fitness * generations)
    
    def calculate_summary_statistics(self, results_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate summary statistics for each configuration.
        
        Args:
            results_df: DataFrame with experimental results
            
        Returns:
            DataFrame with summary statistics
        """
        summary_stats = results_df.groupby(['config_name', 'function_name', 'representation', 'crossover_type']).agg({
            'best_fitness': ['mean', 'std', 'min', 'max', 'median'],
            'final_generation': ['mean', 'std'],
            'fitness_evaluations': ['mean', 'std'],
            'execution_time': ['mean', 'std'],
            'convergence_rate': ['mean', 'std'],
            'success': ['sum', 'mean']  # Count and rate
        }).round(6)
        
        # Flatten column names
        summary_stats.columns = ['_'.join(col).strip() for col in summary_stats.columns]
        summary_stats = summary_stats.reset_index()
        
        return summary_stats
    
    def save_convergence_data(self, results: List[Dict[str, Any]], save_path: str):
        """
        Save convergence histories for plotting.
        
        Args:
            results: List of experiment results
            save_path: Path to save convergence data
        """
        convergence_data = {}
        
        for result in results:
            config_name = result['config_name']
            run_number = result['run_number']
            
            key = f"{config_name}_run_{run_number}"
            convergence_data[key] = {
                'config_name': config_name,
                'run_number': run_number,
                'best_fitness_history': result['convergence_history'],
                'avg_fitness_history': result['avg_fitness_history']
            }
        
        with open(save_path, 'w') as f:
            json.dump(convergence_data, f, indent=2)
