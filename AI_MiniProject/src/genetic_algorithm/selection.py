"""
Selection methods for genetic algorithms.

This module implements various selection strategies for parent selection.
"""

import numpy as np


def tournament_selection(population: np.ndarray, fitness_values: np.ndarray, 
                        tournament_size: int = 3) -> np.ndarray:
    """
    Tournament selection method.
    
    Args:
        population: Population array
        fitness_values: Fitness values for population
        tournament_size: Number of individuals in tournament
        
    Returns:
        Selected individual
    """
    selected_indices = np.random.choice(len(population), tournament_size, replace=False)
    tournament_fitness = fitness_values[selected_indices]
    winner_idx = selected_indices[np.argmin(tournament_fitness)]
    return population[winner_idx].copy()


def roulette_wheel_selection(population: np.ndarray, fitness_values: np.ndarray) -> np.ndarray:
    """
    Roulette wheel selection method.
    
    Args:
        population: Population array
        fitness_values: Fitness values for population
        
    Returns:
        Selected individual
    """
    # Convert fitness to selection probabilities (inverse for minimization)
    max_fitness = np.max(fitness_values)
    adjusted_fitness = max_fitness - fitness_values + 1e-10
    probabilities = adjusted_fitness / np.sum(adjusted_fitness)
    
    selected_idx = np.random.choice(len(population), p=probabilities)
    return population[selected_idx].copy()
