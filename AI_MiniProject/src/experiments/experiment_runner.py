"""
Experiment runner for systematic GA optimization experiments.

This module orchestrates the execution of multiple GA runs
with different configurations and parameter settings.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any
import time
from pathlib import Path

from src.benchmark_functions import create_benchmark_functions
from src.genetic_algorithm import BinaryGA, RealValuedGA
from .config_manager import ConfigManager, GAConfig
from .data_collector import DataCollector


class ExperimentRunner:
    """Runs systematic GA optimization experiments."""
    
    def __init__(self, 
                 runs_per_config: int = 30,
                 max_fitness_evaluations: int = 10000,
                 results_dir: str = "results",
                 random_seed_base: int = 42):
        """
        Initialize experiment runner.
        
        Args:
            runs_per_config: Number of independent runs per configuration
            max_fitness_evaluations: Maximum fitness evaluations per run
            results_dir: Directory to save results
            random_seed_base: Base seed for reproducible results
        """
        self.runs_per_config = runs_per_config
        self.max_fitness_evaluations = max_fitness_evaluations
        self.results_dir = Path(results_dir)
        self.random_seed_base = random_seed_base
        
        # Create results directory
        self.results_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.functions = create_benchmark_functions()
        self.data_collector = DataCollector()
        
    def create_ga_instance(self, config: GAConfig, function) -> Any:
        """
        Create GA instance based on configuration.
        
        Args:
            config: GA configuration
            function: Benchmark function
            
        Returns:
            GA instance
        """
        common_params = {
            'fitness_function': function,
            'domain': function.domain,
            'dimensions': function.dimension,
            'population_size': config.population_size,
            'max_generations': config.max_generations,
            'crossover_rate': config.crossover_rate,
            'mutation_rate': config.mutation_rate,
            'selection_method': config.selection_method,
            'tournament_size': config.tournament_size,
            'elitism_count': config.elitism_count
        }
        
        if config.representation == 'binary':
            return BinaryGA(
                crossover_type=config.crossover_type,
                bits_per_dimension=config.bits_per_dimension,
                **common_params
            )
        elif config.representation == 'real_valued':
            return RealValuedGA(
                crossover_type=config.crossover_type,
                mutation_strength=config.mutation_strength,
                **common_params
            )
        else:
            raise ValueError(f"Unknown representation: {config.representation}")
    
    def run_single_experiment(self, config: GAConfig, run_number: int) -> Dict[str, Any]:
        """
        Run a single GA experiment.
        
        Args:
            config: GA configuration
            run_number: Run identifier
            
        Returns:
            Results dictionary
        """
        function = self.functions[config.function_name]
        
        # Create GA with unique seed
        seed = self.random_seed_base + run_number
        ga = self.create_ga_instance(config, function)
        ga.random_seed = seed
        np.random.seed(seed)
        
        # Run optimization
        start_time = time.time()
        results = ga.run(max_fitness_evaluations=self.max_fitness_evaluations)
        end_time = time.time()
        
        # Compile results
        experiment_results = {
            'config_name': ConfigManager.get_config_name(config),
            'function_name': config.function_name,
            'representation': config.representation,
            'crossover_type': config.crossover_type,
            'run_number': run_number,
            'seed': seed,
            'best_fitness': results['best_fitness'],
            'best_individual': results['best_individual'],
            'final_generation': results['generations_run'],
            'fitness_evaluations': results['fitness_evaluations'],
            'execution_time': end_time - start_time,
            'convergence_history': results['best_fitness_history'],
            'avg_fitness_history': results['avg_fitness_history']
        }
        
        return experiment_results
    
    def run_configuration(self, config: GAConfig) -> List[Dict[str, Any]]:
        """
        Run all experiments for a single configuration.
        
        Args:
            config: GA configuration
            
        Returns:
            List of results for all runs
        """
        config_name = ConfigManager.get_config_name(config)
        print(f"Running {self.runs_per_config} experiments for {config_name}...")
        
        results = []
        for run in range(self.runs_per_config):
            if (run + 1) % 10 == 0:
                print(f"  Completed {run + 1}/{self.runs_per_config} runs")
            
            result = self.run_single_experiment(config, run)
            results.append(result)
        
        print(f"Completed {config_name}")
        return results
    
    def run_all_experiments(self) -> pd.DataFrame:
        """
        Run all experimental configurations.
        
        Returns:
            DataFrame with all results
        """
        print("Starting comprehensive GA optimization experiments")
        print("=" * 50)
        
        all_results = []
        configurations = ConfigManager.create_all_configurations()
        
        total_configs = len(configurations)
        total_runs = total_configs * self.runs_per_config
        
        print(f"Total configurations: {total_configs}")
        print(f"Total runs: {total_runs}")
        print(f"Max evaluations per run: {self.max_fitness_evaluations}")
        print(f"Total evaluations: {total_runs * self.max_fitness_evaluations:,}")
        print()
        
        start_time = time.time()
        
        for i, config in enumerate(configurations):
            print(f"Configuration {i+1}/{total_configs}")
            config_results = self.run_configuration(config)
            all_results.extend(config_results)
            
            # Save intermediate results
            if (i + 1) % 2 == 0:
                self.save_results(all_results, intermediate=True)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        print(f"\nAll experiments completed in {total_time:.2f} seconds")
        print(f"Average time per run: {total_time/total_runs:.2f} seconds")
        
        # Save final results
        results_df = self.save_results(all_results)
        return results_df
    
    def save_results(self, results: List[Dict[str, Any]], 
                    intermediate: bool = False) -> pd.DataFrame:
        """
        Save results to files.
        
        Args:
            results: List of experiment results
            intermediate: Whether this is an intermediate save
            
        Returns:
            DataFrame with results
        """
        # Create DataFrame
        results_df = self.data_collector.create_results_dataframe(results)
        
        # Save to CSV
        suffix = "_intermediate" if intermediate else ""
        csv_path = self.results_dir / f"ga_experiments{suffix}.csv"
        results_df.to_csv(csv_path, index=False)
        
        if not intermediate:
            print(f"Results saved to {csv_path}")
        
        return results_df
