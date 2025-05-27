"""
Main execution script for GA optimization experiments.

This script orchestrates the complete experimental pipeline including
function visualization, GA optimization, and statistical analysis.
"""

import sys
from pathlib import Path

from src.benchmark_functions import create_benchmark_functions
from visualization import FunctionVisualizer
from experiments import ExperimentRunner
from experiments import DataCollector
from analysis import StatisticalAnalyzer


def visualize_functions():
    """Create visualizations for benchmark functions."""
    print("Creating function visualizations...")
    
    functions = create_benchmark_functions()
    visualizer = FunctionVisualizer()
    
    for name, func in functions.items():
        print(f"Visualizing {func.name}...")
        visualizer.plot_function_2d(func, resolution=100)


def run_experiments():
    """Run comprehensive GA optimization experiments."""
    print("Starting GA optimization experiments...")
    
    # Configure experiment
    runner = ExperimentRunner(
        runs_per_config=30,
        max_fitness_evaluations=10000,
        results_dir="results",
        random_seed_base=42
    )
    
    # Run all experiments
    results_df = runner.run_all_experiments()
    
    return results_df


def analyze_results(results_df):
    """Perform statistical analysis of results."""
    print("Performing statistical analysis...")
    
    analyzer = StatisticalAnalyzer(alpha=0.05)
    data_collector = DataCollector()
    
    # Calculate summary statistics
    summary_stats = data_collector.calculate_summary_statistics(results_df)
    summary_path = Path("results") / "summary_statistics.csv"
    summary_stats.to_csv(summary_path, index=False)
    print(f"Summary statistics saved to {summary_path}")
    
    # Analyze configuration differences
    comparison_results = analyzer.analyze_configuration_differences(results_df, 'best_fitness')
    comparison_path = Path("results") / "statistical_comparisons.csv"
    comparison_results.to_csv(comparison_path, index=False)
    print(f"Statistical comparisons saved to {comparison_path}")
    
    # Display top performers
    print("\nTop 5 configurations by mean best fitness:")
    top_configs = summary_stats.nsmallest(5, 'best_fitness_mean')[
        ['config_name', 'function_name', 'representation', 'crossover_type', 'best_fitness_mean', 'best_fitness_std']
    ]
    print(top_configs.to_string(index=False))
    
    return summary_stats, comparison_results


def run_gui():
    """Launch the GUI application."""
    try:
        from gui.main_window import GAOptimizerGUI
        app = GAOptimizerGUI()
        app.run()
    except ImportError:
        print("PySimpleGUI not installed. Install with: pip install PySimpleGUI")
        return False
    except Exception as e:
        print(f"Error launching GUI: {e}")
        return False
    return True


def main():
    """Main execution function."""
    print("GA Benchmark Optimization Project")
    print("=" * 50)
    
    # Check for GUI mode
    if len(sys.argv) > 1 and sys.argv[1].lower() in ['--gui', '-g', 'gui']:
        print("Launching GUI mode...")
        if run_gui():
            return
        else:
            print("Falling back to command line mode...")
    
    # Create results directory
    Path("results").mkdir(exist_ok=True)
    
    # Ask user for mode
    print("\nSelect mode:")
    print("1. Full experimental pipeline (command line)")
    print("2. Function visualization only")
    print("3. Launch GUI (if available)")
    print("4. Exit")
    
    try:
        choice = input("\nEnter choice (1-4): ").strip()
        
        if choice == "1":
            # Step 1: Visualize functions
            visualize_functions()
            
            # Step 2: Run experiments
            results_df = run_experiments()
            
            # Step 3: Analyze results
            summary_stats, comparison_results = analyze_results(results_df)
            
            print("Summary stats:", summary_stats)
            print("Comparison results:", comparison_results)
            
            print("\nExperiment pipeline completed successfully!")
            print(f"Results saved in 'results/' directory")
            
        elif choice == "2":
            visualize_functions()
            print("Function visualization completed!")
            
        elif choice == "3":
            if not run_gui():
                print("GUI not available.")
                
        elif choice == "4":
            print("Goodbye!")
            return
            
        else:
            print("Invalid choice. Exiting.")
            
    except KeyboardInterrupt:
        print("\nOperation interrupted by user")
    except Exception as e:
        print(f"\nError occurred: {e}")
        raise


if __name__ == "__main__":
    main()
