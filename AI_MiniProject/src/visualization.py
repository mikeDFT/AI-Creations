"""
Visualization module for benchmark functions and GA optimization results.

This module provides functions to create 2D contour plots and 3D surface plots
for the benchmark functions, as well as visualization of GA optimization progress.
"""

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Optional
from src.benchmark_functions import BenchmarkFunction


class FunctionVisualizer:
    """Class for visualizing benchmark functions and optimization results."""
    
    def __init__(self, figsize: Tuple[int, int] = (12, 5)):
        """
        Initialize the visualizer.
        
        Args:
            figsize: Default figure size for plots
        """
        self.figsize = figsize
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def plot_function_2d(self, func: BenchmarkFunction, resolution: int = 100, 
                        show_optimum: bool = True, save_path: Optional[str] = None) -> None:
        """
        Create 2D contour and 3D surface plots for a benchmark function.
        
        Args:
            func: Benchmark function to visualize
            resolution: Grid resolution for plotting
            show_optimum: Whether to mark the global optimum
            save_path: Path to save the figure (optional)
        """
        if func.dimension != 2:
            raise ValueError("This method only supports 2D functions")
        
        # Create grid for evaluation
        x_range = np.linspace(func.domain[0], func.domain[1], resolution)
        y_range = np.linspace(func.domain[0], func.domain[1], resolution)
        X, Y = np.meshgrid(x_range, y_range)
        
        # Evaluate function on grid
        Z = np.zeros_like(X)
        for i in range(resolution):
            for j in range(resolution):
                Z[i, j] = func([X[i, j], Y[i, j]])
        
        # Create subplot figure
        fig = plt.figure(figsize=(15, 6))
          # 2D Contour plot
        ax1 = fig.add_subplot(121)
        # Create filled contours
        contourf = ax1.contourf(X, Y, Z, levels=50, cmap='viridis', alpha=0.9)
        # Add contour lines without labels
        contour = ax1.contour(X, Y, Z, levels=15, colors='white', alpha=0.8, linewidths=0.8)
        # Remove any inline labels (this prevents numbers from showing on contours)
        
        if show_optimum:
            ax1.plot(func.global_optimum[0], func.global_optimum[1], 
                    'r*', markersize=15, label=f'Global Optimum: {func.global_minimum:.3f}')
            ax1.legend()
        
        ax1.set_xlabel('x1')
        ax1.set_ylabel('x2')
        ax1.set_title(f'{func.name} - 2D Contour Plot')
        ax1.grid(True, alpha=0.3)
        plt.colorbar(contourf, ax=ax1, label='Function Value')
        
        # 3D Surface plot
        ax2 = fig.add_subplot(122, projection='3d')
        surface = ax2.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8, 
                                  linewidth=0, antialiased=True)
        
        if show_optimum:
            ax2.scatter(func.global_optimum[0], func.global_optimum[1], func.global_minimum,
                       color='red', s=100, label=f'Global Optimum: {func.global_minimum:.3f}')
            ax2.legend()
        
        ax2.set_xlabel('x1')
        ax2.set_ylabel('x2')
        ax2.set_zlabel('f(x1, x2)')
        ax2.set_title(f'{func.name} - 3D Surface Plot')
        plt.colorbar(surface, ax=ax2, shrink=0.5, label='Function Value')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            # Save to default location if no path specified
            plt.savefig(f'{func.name.replace(" ", "_").lower()}_plot.png', dpi=300, bbox_inches='tight')
        
        print(f"Plot saved for {func.name}")
        plt.show()
    
    def plot_optimization_progress(self, best_fitness_history: List[float], 
                                 avg_fitness_history: List[float],
                                 title: str = "GA Optimization Progress",
                                 save_path: Optional[str] = None) -> None:
        """
        Plot the optimization progress showing best and average fitness over generations.
        
        Args:
            best_fitness_history: List of best fitness values per generation
            avg_fitness_history: List of average fitness values per generation
            title: Plot title
            save_path: Path to save the figure (optional)
        """
        plt.figure(figsize=self.figsize)
        
        generations = range(len(best_fitness_history))
        
        plt.plot(generations, best_fitness_history, 'b-', linewidth=2, 
                label='Best Fitness', marker='o', markersize=3)
        plt.plot(generations, avg_fitness_history, 'r--', linewidth=2, 
                label='Average Fitness', marker='s', markersize=3)
        
        plt.xlabel('Generation')
        plt.ylabel('Fitness Value')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale('log')  # Log scale often better for optimization plots
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_population_evolution(self, populations: List[np.ndarray], 
                                func: BenchmarkFunction,
                                generations_to_show: List[int] = None,
                                save_path: Optional[str] = None) -> None:
        """
        Plot the evolution of population over selected generations (for 2D functions).
        
        Args:
            populations: List of population arrays for each generation
            func: The benchmark function being optimized
            generations_to_show: Which generations to display (default: first, middle, last)
            save_path: Path to save the figure (optional)
        """
        if func.dimension != 2:
            raise ValueError("Population evolution plot only supports 2D functions")
        
        if generations_to_show is None:
            n_gens = len(populations)
            generations_to_show = [0, n_gens//2, n_gens-1] if n_gens > 2 else list(range(n_gens))
        
        n_plots = len(generations_to_show)
        fig, axes = plt.subplots(1, n_plots, figsize=(5*n_plots, 5))
        
        if n_plots == 1:
            axes = [axes]
        
        # Create background contour
        resolution = 50
        x_range = np.linspace(func.domain[0], func.domain[1], resolution)
        y_range = np.linspace(func.domain[0], func.domain[1], resolution)
        X, Y = np.meshgrid(x_range, y_range)
        Z = np.zeros_like(X)
        for i in range(resolution):
            for j in range(resolution):
                Z[i, j] = func([X[i, j], Y[i, j]])
        
        for idx, gen in enumerate(generations_to_show):
            ax = axes[idx]
            
            # Plot contour background
            contour = ax.contour(X, Y, Z, levels=15, cmap='viridis', alpha=0.3)
            
            # Plot population
            pop = populations[gen]
            ax.scatter(pop[:, 0], pop[:, 1], c='red', alpha=0.7, s=30, 
                      label=f'Population (Gen {gen})')
            
            # Mark global optimum
            ax.plot(func.global_optimum[0], func.global_optimum[1], 
                   'g*', markersize=15, label='Global Optimum')
            
            ax.set_xlabel('x1')
            ax.set_ylabel('x2')
            ax.set_title(f'Generation {gen}')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_xlim(func.domain)
            ax.set_ylim(func.domain)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_convergence_comparison(self, results_dict: dict, 
                                  title: str = "GA Convergence Comparison",
                                  save_path: Optional[str] = None) -> None:
        """
        Compare convergence of different GA configurations.
        
        Args:
            results_dict: Dictionary with format {config_name: best_fitness_history}
            title: Plot title
            save_path: Path to save the figure (optional)
        """
        plt.figure(figsize=(12, 8))
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(results_dict)))
        
        for i, (config_name, fitness_history) in enumerate(results_dict.items()):
            generations = range(len(fitness_history))
            plt.plot(generations, fitness_history, linewidth=2, 
                    color=colors[i], label=config_name, marker='o', markersize=2)
        
        plt.xlabel('Generation')
        plt.ylabel('Best Fitness Value')
        plt.title(title)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_statistical_results(self, results_df, metric: str = 'best_fitness',
                               save_path: Optional[str] = None) -> None:
        """
        Create box plots and violin plots for statistical analysis of results.
        
        Args:
            results_df: DataFrame with columns ['configuration', 'run', metric]
            metric: Column name to plot
            save_path: Path to save the figure (optional)
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Box plot
        sns.boxplot(data=results_df, x='configuration', y=metric, ax=ax1)
        ax1.set_title(f'Box Plot - {metric}')
        ax1.tick_params(axis='x', rotation=45)
        
        # Violin plot
        sns.violinplot(data=results_df, x='configuration', y=metric, ax=ax2)
        ax2.set_title(f'Violin Plot - {metric}')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()

# example usage and testing
if __name__ == "__main__":
    from src.benchmark_functions import create_benchmark_functions
    
    # Create visualizer
    visualizer = FunctionVisualizer()
    
    # Create and visualize benchmark functions
    functions = create_benchmark_functions()
    
    for name, func in functions.items():
        print(f"Visualizing {func.name}...")
        visualizer.plot_function_2d(func, resolution=100)
