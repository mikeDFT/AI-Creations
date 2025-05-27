"""
Configuration manager for GA experiments.

This module handles the creation and management of different
GA configurations for systematic experimentation.
"""

from typing import Dict, List, Any
from dataclasses import dataclass


@dataclass
class GAConfig:
    """Configuration for a single GA experiment."""
    function_name: str
    representation: str  # 'binary' or 'real_valued'
    crossover_type: str
    population_size: int = 50
    max_generations: int = 100
    crossover_rate: float = 0.8
    mutation_rate: float = 0.1
    selection_method: str = 'tournament'
    tournament_size: int = 3
    elitism_count: int = 1
    bits_per_dimension: int = 16  # For binary representation
    mutation_strength: float = 0.1  # For real-valued representation


class ConfigManager:
    """Manages experimental configurations."""
    
    @staticmethod
    def create_all_configurations() -> List[GAConfig]:
        """
        Create all required experimental configurations.
        
        Returns:
            List of GAConfig objects for all experiments
        """
        functions = ['ackley', 'rastrigin']
        configurations = []
        
        for function_name in functions:
            # Binary representations
            for crossover_type in ['one_point', 'two_point']:
                config = GAConfig(
                    function_name=function_name,
                    representation='binary',
                    crossover_type=crossover_type
                )
                configurations.append(config)
            
            # Real-valued representations
            for crossover_type in ['arithmetic', 'blx_alpha']:
                config = GAConfig(
                    function_name=function_name,
                    representation='real_valued',
                    crossover_type=crossover_type
                )
                configurations.append(config)
        
        return configurations
    
    @staticmethod
    def config_to_dict(config: GAConfig) -> Dict[str, Any]:
        """Convert GAConfig to dictionary."""
        return {
            'function_name': config.function_name,
            'representation': config.representation,
            'crossover_type': config.crossover_type,
            'population_size': config.population_size,
            'max_generations': config.max_generations,
            'crossover_rate': config.crossover_rate,
            'mutation_rate': config.mutation_rate,
            'selection_method': config.selection_method,
            'tournament_size': config.tournament_size,
            'elitism_count': config.elitism_count,
            'bits_per_dimension': config.bits_per_dimension,
            'mutation_strength': config.mutation_strength
        }
    
    @staticmethod
    def get_config_name(config: GAConfig) -> str:
        """Generate a descriptive name for the configuration."""
        return f"{config.function_name}_{config.representation}_{config.crossover_type}"
