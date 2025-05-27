"""
Real-valued Genetic Algorithm implementation.

This module implements a GA with real-valued chromosome representation
and includes arithmetic and BLX-α crossover operators.
"""

import numpy as np
from typing import Tuple, Callable
from .base_ga import GeneticAlgorithm
from .crossover import arithmetic_crossover, blx_alpha_crossover
from .mutation import gaussian_mutation


class RealValuedGA(GeneticAlgorithm):
    """Genetic Algorithm with real-valued representation."""
    
    def __init__(self, fitness_function: Callable, domain: Tuple[float, float],
                 dimensions: int = 2, crossover_type: str = 'arithmetic',
                 mutation_strength: float = 0.1, **kwargs):
        """
        Initialize Real-valued GA.
        
        Args:
            fitness_function: Function to optimize
            domain: Domain bounds (min, max)
            dimensions: Number of dimensions
            crossover_type: 'arithmetic' or 'blx_alpha'
            mutation_strength: Standard deviation for Gaussian mutation
            **kwargs: Additional parameters for base GA
        """
        super().__init__(fitness_function, **kwargs)
        self.domain = domain
        self.dimensions = dimensions
        self.crossover_type = crossover_type
        self.mutation_strength = mutation_strength
    
    def initialize_population(self) -> np.ndarray:
        """Initialize random real-valued population."""
        return np.random.uniform(self.domain[0], self.domain[1], 
                               (self.population_size, self.dimensions))
    
    def decode_individual(self, individual: np.ndarray) -> np.ndarray:
        """Real-valued individuals don't need decoding."""
        return individual.copy()
    
    def crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Perform crossover based on selected type."""
        if self.crossover_type == 'arithmetic':
            return arithmetic_crossover(parent1, parent2)
        elif self.crossover_type == 'blx_alpha':
            return blx_alpha_crossover(parent1, parent2, alpha=0.5)
        else:
            raise ValueError(f"Unknown crossover type: {self.crossover_type}")
    
    def mutate(self, individual: np.ndarray) -> np.ndarray:
        """Mutate real-valued individual."""
        return gaussian_mutation(individual, self.mutation_strength, self.domain)
