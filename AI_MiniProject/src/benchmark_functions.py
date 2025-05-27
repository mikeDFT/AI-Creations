"""
Benchmark optimization functions for Genetic Algorithm testing.

This module implements the Ackley and Rastrigin functions
multimodal benchmark functions for testing optimization algorithms.
"""

import numpy as np
from typing import Union, Tuple


class BenchmarkFunction:
    """Base class for benchmark optimization functions."""
    
    def __init__(self, name: str, dimension: int, domain: Tuple[float, float]):
        """
        Initialize a benchmark function.
        
        Args:
            name: Name of the function
            dimension: Dimensionality of the function
            domain: Tuple representing the domain (min_val, max_val)
        """
        self.name = name
        self.dimension = dimension
        self.domain = domain
        self.global_minimum = 0.0  # Both Ackley and Rastrigin have global minimum at 0
        self.global_optimum = np.zeros(dimension)  # Global optimum at origin
    
    def __call__(self, x: Union[np.ndarray, list]) -> float:
        """Evaluate the function at point x."""
        raise NotImplementedError("Subclasses must implement __call__ method")
    
    def is_in_domain(self, x: Union[np.ndarray, list]) -> bool:
        """Check if point x is within the function's domain."""
        x = np.array(x)
        return np.all((x >= self.domain[0]) & (x <= self.domain[1]))


class AckleyFunction(BenchmarkFunction):
    """
    Ackley function implementation.
    
    The Ackley function is a multimodal function with many local minima.
    Global minimum: f(0, 0) = 0
    Domain: typically [-32.768, 32.768] for each dimension
    
    Formula: f(x) = -a * exp(-b * sqrt(1/d * sum(x_i^2))) - exp(1/d * sum(cos(c * x_i))) + a + exp(1)
    where a=20, b=0.2, c=2π
    """
    
    def __init__(self, dimension: int = 2, domain: Tuple[float, float] = (-32.768, 32.768)):
        """
        Initialize Ackley function.
        
        Args:
            dimension: Number of dimensions (default: 2)
            domain: Domain bounds (default: (-32.768, 32.768))
        """
        super().__init__(f"Ackley {dimension}D", dimension, domain)
        self.a = 20.0
        self.b = 0.2
        self.c = 2.0 * np.pi
    
    def __call__(self, x: Union[np.ndarray, list]) -> float:
        """
        Evaluate Ackley function at point x.
        
        Args:
            x: Input point (array-like)
            
        Returns:
            Function value at x
        """
        x = np.array(x)
        if len(x) != self.dimension:
            raise ValueError(f"Expected {self.dimension} dimensions, got {len(x)}")
        
        # Calculate the three terms of the Ackley function
        term1 = -self.a * np.exp(-self.b * np.sqrt(np.mean(x**2)))
        term2 = -np.exp(np.mean(np.cos(self.c * x)))
        term3 = self.a + np.exp(1)
        
        return term1 + term2 + term3


class RastriginFunction(BenchmarkFunction):
    """
    Rastrigin function implementation.
    
    The Rastrigin function is a highly multimodal function with many local minima.
    Global minimum: f(0, 0) = 0
    Domain: typically [-5.12, 5.12] for each dimension
    
    Formula: f(x) = A * d + sum(x_i^2 - A * cos(2π * x_i))
    where A=10
    """
    
    def __init__(self, dimension: int = 2, domain: Tuple[float, float] = (-5.12, 5.12)):
        """
        Initialize Rastrigin function.
        
        Args:
            dimension: Number of dimensions (default: 2)
            domain: Domain bounds (default: (-5.12, 5.12))
        """
        super().__init__(f"Rastrigin {dimension}D", dimension, domain)
        self.A = 10.0
    
    def __call__(self, x: Union[np.ndarray, list]) -> float:
        """
        Evaluate Rastrigin function at point x.
        
        Args:
            x: Input point (array-like)
            
        Returns:
            Function value at x
        """
        x = np.array(x)
        if len(x) != self.dimension:
            raise ValueError(f"Expected {self.dimension} dimensions, got {len(x)}")
        
        # Calculate Rastrigin function
        return self.A * self.dimension + np.sum(x**2 - self.A * np.cos(2 * np.pi * x))


def create_benchmark_functions() -> dict:
    """
    Create instances of all benchmark functions.
    
    Returns:
        Dictionary mapping function names to function instances
    """
    functions = {
        'ackley': AckleyFunction(dimension=2),
        'rastrigin': RastriginFunction(dimension=2)
    }
    
    return functions


# example usage and testing
if __name__ == "__main__":
    # Test the functions at known points
    functions = create_benchmark_functions()
    
    print("Testing benchmark functions:")
    print("=" * 40)
    
    for name, func in functions.items():
        print(f"\n{func.name}:")
        print(f"Domain: {func.domain}")
        print(f"Global optimum: {func.global_optimum}")
        print(f"Global minimum: {func.global_minimum}")
        
        # Test at global optimum
        value_at_optimum = func(func.global_optimum)
        print(f"f(global_optimum) = {value_at_optimum:.6f}")
        
        # Test at a random point
        np.random.seed(42)
        random_point = np.random.uniform(func.domain[0], func.domain[1], func.dimension)
        value_at_random = func(random_point)
        print(f"f({random_point}) = {value_at_random:.6f}")
    
    # Test with visualization if available
    try:
        from visualization import FunctionVisualizer
        print("\n" + "=" * 40)
        print("Creating visualizations...")
        visualizer = FunctionVisualizer()
        for name, func in functions.items():
            visualizer.plot_function_2d(func, resolution=50)
    except ImportError:
        print("Visualization module not available")
