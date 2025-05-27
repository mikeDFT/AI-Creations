"""
Binary Genetic Algorithm implementation.

This module implements a GA with binary chromosome representation
and includes 1-point and 2-point crossover operators.
"""

import numpy as np
from typing import Tuple, Callable
from .base_ga import GeneticAlgorithm
from .crossover import one_point_crossover, two_point_crossover
from .mutation import binary_mutation


class BinaryGA(GeneticAlgorithm):
    """Genetic Algorithm with binary representation."""
    
    def __init__(self, fitness_function: Callable, domain: Tuple[float, float],
                 dimensions: int = 2, bits_per_dimension: int = 16,
                 crossover_type: str = 'one_point', **kwargs):
        """
        Initialize Binary GA.
        
        Args:
            fitness_function: Function to optimize
            domain: Domain bounds (min, max)
            dimensions: Number of dimensions
            bits_per_dimension: Number of bits per dimension
            crossover_type: 'one_point' or 'two_point'
            **kwargs: Additional parameters for base GA
        """
        super().__init__(fitness_function, **kwargs)
        self.domain = domain
        self.dimensions = dimensions
        self.bits_per_dimension = bits_per_dimension
        self.chromosome_length = dimensions * bits_per_dimension
        self.crossover_type = crossover_type
        
        # Calculate precision
        self.precision = (domain[1] - domain[0]) / (2**bits_per_dimension - 1)
    
    def initialize_population(self) -> np.ndarray:
        """Initialize random binary population."""
        return np.random.randint(0, 2, (self.population_size, self.chromosome_length))
    
    def decode_individual(self, individual: np.ndarray) -> np.ndarray:
        """Decode binary individual to real values."""
        real_values = np.zeros(self.dimensions)
        
        for i in range(self.dimensions):
            start_bit = i * self.bits_per_dimension
            end_bit = start_bit + self.bits_per_dimension
            bits = individual[start_bit:end_bit]
            
            # Convert binary to decimal
            decimal_value = 0
            for j, bit in enumerate(bits):
                decimal_value += bit * (2 ** (self.bits_per_dimension - 1 - j))
            
            # Scale to domain
            real_values[i] = self.domain[0] + decimal_value * self.precision
        
        return real_values
    
    def crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Perform crossover based on selected type."""
        if self.crossover_type == 'one_point':
            return one_point_crossover(parent1, parent2)
        elif self.crossover_type == 'two_point':
            return two_point_crossover(parent1, parent2)
        else:
            raise ValueError(f"Unknown crossover type: {self.crossover_type}")
    
    def mutate(self, individual: np.ndarray) -> np.ndarray:
        """Mutate binary individual."""
        return binary_mutation(individual, self.mutation_rate)
