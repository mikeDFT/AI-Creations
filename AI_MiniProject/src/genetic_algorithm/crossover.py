"""
Crossover operators for genetic algorithms.

This module implements various crossover operators for both binary
and real-valued representations.
"""

import numpy as np
from typing import Tuple


def one_point_crossover(parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    One-point crossover for binary representations.
    
    Args:
        parent1, parent2: Parent chromosomes
        
    Returns:
        Tuple of two offspring
    """
    if len(parent1) != len(parent2):
        raise ValueError("Parents must have same length")
    
    crossover_point = np.random.randint(1, len(parent1))
    
    offspring1 = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
    offspring2 = np.concatenate([parent2[:crossover_point], parent1[crossover_point:]])
    
    return offspring1, offspring2


def two_point_crossover(parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Two-point crossover for binary representations.
    
    Args:
        parent1, parent2: Parent chromosomes
        
    Returns:
        Tuple of two offspring
    """
    if len(parent1) != len(parent2):
        raise ValueError("Parents must have same length")
    
    point1, point2 = sorted(np.random.choice(range(1, len(parent1)), 2, replace=False))
    
    offspring1 = np.concatenate([parent1[:point1], parent2[point1:point2], parent1[point2:]])
    offspring2 = np.concatenate([parent2[:point1], parent1[point1:point2], parent2[point2:]])
    
    return offspring1, offspring2


def arithmetic_crossover(parent1: np.ndarray, parent2: np.ndarray, alpha: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Arithmetic crossover for real-valued representations.
    
    Args:
        parent1, parent2: Parent chromosomes
        alpha: Blending parameter
        
    Returns:
        Tuple of two offspring
    """
    offspring1 = alpha * parent1 + (1 - alpha) * parent2
    offspring2 = (1 - alpha) * parent1 + alpha * parent2
    
    return offspring1, offspring2


def blx_alpha_crossover(parent1: np.ndarray, parent2: np.ndarray, alpha: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    """
    BLX-α crossover for real-valued representations.
    
    Args:
        parent1, parent2: Parent chromosomes
        alpha: Extension parameter
        
    Returns:
        Tuple of two offspring
    """
    offspring1 = np.zeros_like(parent1)
    offspring2 = np.zeros_like(parent2)
    
    for i in range(len(parent1)):
        min_val = min(parent1[i], parent2[i])
        max_val = max(parent1[i], parent2[i])
        range_val = max_val - min_val
        
        lower_bound = min_val - alpha * range_val
        upper_bound = max_val + alpha * range_val
        
        offspring1[i] = np.random.uniform(lower_bound, upper_bound)
        offspring2[i] = np.random.uniform(lower_bound, upper_bound)
    
    return offspring1, offspring2
