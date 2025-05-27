"""
Mutation operators for genetic algorithms.

This module implements various mutation operators for both binary
and real-valued representations.
"""

import numpy as np


def binary_mutation(individual: np.ndarray, mutation_rate: float = 0.01) -> np.ndarray:
    """
    Bit-flip mutation for binary representations.
    
    Args:
        individual: Binary chromosome
        mutation_rate: Probability of flipping each bit
        
    Returns:
        Mutated individual
    """
    mutated = individual.copy()
    mutation_mask = np.random.random(len(individual)) < mutation_rate
    mutated[mutation_mask] = 1 - mutated[mutation_mask]
    return mutated


def gaussian_mutation(individual: np.ndarray, mutation_strength: float = 0.1, 
                     bounds: tuple = (-5, 5)) -> np.ndarray:
    """
    Gaussian mutation for real-valued representations.
    
    Args:
        individual: Real-valued chromosome
        mutation_strength: Standard deviation of Gaussian noise
        bounds: Domain bounds (min, max)
        
    Returns:
        Mutated individual
    """
    mutated = individual + np.random.normal(0, mutation_strength, len(individual))
    # Clip to bounds
    mutated = np.clip(mutated, bounds[0], bounds[1])
    return mutated


def uniform_mutation(individual: np.ndarray, mutation_rate: float = 0.1, 
                    bounds: tuple = (-5, 5)) -> np.ndarray:
    """
    Uniform mutation for real-valued representations.
    
    Args:
        individual: Real-valued chromosome
        mutation_rate: Probability of mutating each gene
        bounds: Domain bounds (min, max)
        
    Returns:
        Mutated individual
    """
    mutated = individual.copy()
    mutation_mask = np.random.random(len(individual)) < mutation_rate
    mutated[mutation_mask] = np.random.uniform(bounds[0], bounds[1], np.sum(mutation_mask))
    return mutated
