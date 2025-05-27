"""
Base Genetic Algorithm implementation.

This module provides the abstract base class for genetic algorithms
with common functionality and interface definitions.
"""

import numpy as np
from typing import List, Tuple, Callable, Optional, Dict, Any
from abc import ABC, abstractmethod
import time


class GeneticAlgorithm(ABC):
    """Abstract base class for genetic algorithms."""
    
    def __init__(self, 
                 fitness_function: Callable,
                 population_size: int = 50,
                 max_generations: int = 100,
                 crossover_rate: float = 0.8,
                 mutation_rate: float = 0.1,
                 selection_method: str = 'tournament',
                 tournament_size: int = 3,
                 elitism_count: int = 1,
                 random_seed: Optional[int] = None):
        """
        Initialize the genetic algorithm.
        
        Args:
            fitness_function: Function to optimize (minimize)
            population_size: Number of individuals in population
            max_generations: Maximum number of generations
            crossover_rate: Probability of crossover
            mutation_rate: Probability of mutation
            selection_method: Selection method ('tournament' or 'roulette')
            tournament_size: Size for tournament selection
            elitism_count: Number of best individuals to preserve
            random_seed: Random seed for reproducibility
        """
        self.fitness_function = fitness_function
        self.population_size = population_size
        self.max_generations = max_generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.selection_method = selection_method
        self.tournament_size = tournament_size
        self.elitism_count = elitism_count
        
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # Tracking variables
        self.generation = 0
        self.population = None
        self.fitness_values = None
        self.best_fitness_history = []
        self.avg_fitness_history = []
        self.best_individual = None
        self.best_fitness = float('inf')
        self.fitness_evaluations = 0
        
    @abstractmethod
    def initialize_population(self) -> np.ndarray:
        """Initialize the population."""
        pass
    
    @abstractmethod
    def crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Perform crossover between two parents."""
        pass
    
    @abstractmethod
    def mutate(self, individual: np.ndarray) -> np.ndarray:
        """Mutate an individual."""
        pass
    
    @abstractmethod
    def decode_individual(self, individual: np.ndarray) -> np.ndarray:
        """Decode individual to real values for fitness evaluation."""
        pass
    
    def evaluate_fitness(self, population: np.ndarray) -> np.ndarray:
        """Evaluate fitness for entire population."""
        fitness_values = np.zeros(len(population))
        for i, individual in enumerate(population):
            decoded = self.decode_individual(individual)
            fitness_values[i] = self.fitness_function(decoded)
            self.fitness_evaluations += 1
        return fitness_values
    
    def tournament_selection(self, population: np.ndarray, fitness_values: np.ndarray) -> np.ndarray:
        """Tournament selection method."""
        selected_indices = np.random.choice(len(population), self.tournament_size, replace=False)
        tournament_fitness = fitness_values[selected_indices]
        winner_idx = selected_indices[np.argmin(tournament_fitness)]
        return population[winner_idx].copy()
    
    def roulette_wheel_selection(self, population: np.ndarray, fitness_values: np.ndarray) -> np.ndarray:
        """Roulette wheel selection method."""
        # Convert fitness to selection probabilities (inverse for minimization)
        max_fitness = np.max(fitness_values)
        adjusted_fitness = max_fitness - fitness_values + 1e-10
        probabilities = adjusted_fitness / np.sum(adjusted_fitness)
        
        selected_idx = np.random.choice(len(population), p=probabilities)
        return population[selected_idx].copy()
    
    def select_parent(self, population: np.ndarray, fitness_values: np.ndarray) -> np.ndarray:
        """Select a parent based on selection method."""
        if self.selection_method == 'tournament':
            return self.tournament_selection(population, fitness_values)
        elif self.selection_method == 'roulette':
            return self.roulette_wheel_selection(population, fitness_values)
        else:
            raise ValueError(f"Unknown selection method: {self.selection_method}")
    
    def evolve_generation(self) -> None:
        """Evolve one generation."""
        new_population = []
        
        # Elitism: keep best individuals
        if self.elitism_count > 0:
            elite_indices = np.argsort(self.fitness_values)[:self.elitism_count]
            for idx in elite_indices:
                new_population.append(self.population[idx].copy())
        
        # Generate offspring
        while len(new_population) < self.population_size:
            parent1 = self.select_parent(self.population, self.fitness_values)
            parent2 = self.select_parent(self.population, self.fitness_values)
            
            # Crossover
            if np.random.random() < self.crossover_rate:
                offspring1, offspring2 = self.crossover(parent1, parent2)
            else:
                offspring1, offspring2 = parent1.copy(), parent2.copy()
            
            # Mutation
            if np.random.random() < self.mutation_rate:
                offspring1 = self.mutate(offspring1)
            if np.random.random() < self.mutation_rate:
                offspring2 = self.mutate(offspring2)
            
            new_population.extend([offspring1, offspring2])
        
        # Trim to exact population size
        self.population = np.array(new_population[:self.population_size])
    
    def run(self, max_fitness_evaluations: Optional[int] = None) -> Dict[str, Any]:
        """
        Run the genetic algorithm.
        
        Args:
            max_fitness_evaluations: Maximum number of fitness evaluations
            
        Returns:
            Dictionary with optimization results
        """
        start_time = time.time()
        
        # Initialize population
        self.population = self.initialize_population()
        self.fitness_values = self.evaluate_fitness(self.population)
        
        # Track initial best
        best_idx = np.argmin(self.fitness_values)
        self.best_fitness = self.fitness_values[best_idx]
        self.best_individual = self.decode_individual(self.population[best_idx])
        
        # Evolution loop
        for generation in range(self.max_generations):
            self.generation = generation
            
            # Check termination criteria
            if max_fitness_evaluations and self.fitness_evaluations >= max_fitness_evaluations:
                break
            
            # Record statistics
            self.best_fitness_history.append(np.min(self.fitness_values))
            self.avg_fitness_history.append(np.mean(self.fitness_values))
            
            # Update best individual
            current_best_idx = np.argmin(self.fitness_values)
            current_best_fitness = self.fitness_values[current_best_idx]
            if current_best_fitness < self.best_fitness:
                self.best_fitness = current_best_fitness
                self.best_individual = self.decode_individual(self.population[current_best_idx])
            
            # Evolve population
            self.evolve_generation()
            self.fitness_values = self.evaluate_fitness(self.population)
        
        end_time = time.time()
        
        return {
            'best_fitness': self.best_fitness,
            'best_individual': self.best_individual,
            'best_fitness_history': self.best_fitness_history,
            'avg_fitness_history': self.avg_fitness_history,
            'generations_run': self.generation + 1,
            'fitness_evaluations': self.fitness_evaluations,
            'execution_time': end_time - start_time
        }
