"""
Main GUI window for GA optimization experiments.

This module provides the primary interface for configuring and running
genetic algorithm experiments with real-time progress tracking.
"""

import PySimpleGUI as sg
import threading
import queue
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

from src.benchmark_functions import create_benchmark_functions
from src.genetic_algorithm import BinaryGA, RealValuedGA
from src.experiments import ExperimentRunner, ConfigManager, DataCollector
from src.analysis import StatisticalAnalyzer
from src.visualization import FunctionVisualizer


class GAOptimizerGUI:
    """Main GUI application for GA optimization experiments."""
    
    def __init__(self):
        """Initialize the GUI application."""
        # Set theme - use try/except for compatibility
        try:
            sg.theme('DarkBlue3')
        except AttributeError:
            try:
                sg.change_look_and_feel('DarkBlue3')
            except:
                pass  # Use default theme if neither works
        
        # Initialize components
        self.functions = create_benchmark_functions()
        self.visualizer = FunctionVisualizer()
        self.data_collector = DataCollector()
        self.analyzer = StatisticalAnalyzer()
        
        # GUI state
        self.window = None
        self.experiment_thread = None
        self.results_queue = queue.Queue()
        self.current_results = None
        self.current_canvas_agg = None  # Track current matplotlib canvas
        
        # Create results directory
        Path("results").mkdir(exist_ok=True)
    
    def create_layout(self):
        """Create the main window layout."""
        # Function visualization tab
        viz_tab = [
            [sg.Text("Function Visualization", font=('Arial', 16, 'bold'))],
            [sg.Frame("Function Settings", [
                [sg.Text("Select Function:"),
                 sg.Combo(['Ackley', 'Rastrigin'], default_value='Ackley', key='-VIZ_FUNC-'),
                 sg.Button("Visualize Function", key='-VISUALIZE-')],
                [sg.Text("Resolution:"),
                 sg.Slider(range=(50, 200), default_value=100, orientation='h', key='-RESOLUTION-')],
                [sg.Text("Plot Size:"),
                 sg.Combo(['Small (12x5)', 'Medium (15x6)', 'Large (18x8)', 'Extra Large (20x10)'], 
                         default_value='Large (18x8)', key='-VIZ_SIZE-')]
            ])],
            [sg.Button("Save Plot", key='-SAVE_VIZ-'), sg.Button("Clear Plot", key='-CLEAR_VIZ-')],
            [sg.Canvas(key='-VIZ_CANVAS-', size=(1200, 600))]  # Increased canvas size
        ]
        
        # Single experiment tab
        single_exp_tab = [
            [sg.Text("Single Experiment Configuration", font=('Arial', 16, 'bold'))],
            [sg.Frame("Function Selection", [
                [sg.Radio("Ackley", "FUNC", default=True, key='-FUNC_ACKLEY-'),
                 sg.Radio("Rastrigin", "FUNC", key='-FUNC_RASTRIGIN-')]
            ])],
            [sg.Frame("GA Configuration", [
                [sg.Text("Representation:"),
                 sg.Radio("Binary", "REP", default=True, key='-REP_BINARY-'),
                 sg.Radio("Real-valued", "REP", key='-REP_REAL-')],
                [sg.Text("Crossover:"),
                 sg.Combo(['one_point', 'two_point', 'arithmetic', 'blx_alpha'],
                         default_value='one_point', key='-CROSSOVER-')],
                [sg.Text("Population Size:"),
                 sg.Spin([i for i in range(10, 201, 10)], initial_value=50, key='-POP_SIZE-')],
                [sg.Text("Max Generations:"),
                 sg.Spin([i for i in range(50, 501, 50)], initial_value=100, key='-MAX_GEN-')],
                [sg.Text("Crossover Rate:"),
                 sg.Slider(range=(0.0, 1.0), default_value=0.8, resolution=0.1,
                          orientation='h', key='-CROSS_RATE-')],
                [sg.Text("Mutation Rate:"),
                 sg.Slider(range=(0.0, 1.0), default_value=0.1, resolution=0.01,
                          orientation='h', key='-MUT_RATE-')]
            ])],
            [sg.Button("Run Single Experiment", key='-RUN_SINGLE-'),
             sg.Button("Clear Results", key='-CLEAR_SINGLE-')],
            [sg.Multiline(size=(100, 25), key='-SINGLE_OUTPUT-', disabled=True)]  # Increased size
        ]
        
        # Batch experiments tab
        batch_exp_tab = [
            [sg.Text("Batch Experiments", font=('Arial', 16, 'bold'))],
            [sg.Frame("Experiment Settings", [
                [sg.Text("Runs per Configuration:"),
                 sg.Spin([i for i in range(5, 51, 5)], initial_value=30, key='-RUNS_PER_CONFIG-')],
                [sg.Text("Max Fitness Evaluations:"),
                 sg.Spin([i for i in range(1000, 21000, 1000)], initial_value=10000, key='-MAX_EVALS-')],
                [sg.Checkbox("Include Binary Representations", default=True, key='-INCLUDE_BINARY-')],
                [sg.Checkbox("Include Real-valued Representations", default=True, key='-INCLUDE_REAL-')],
                [sg.Checkbox("Include Ackley Function", default=True, key='-INCLUDE_ACKLEY-')],
                [sg.Checkbox("Include Rastrigin Function", default=True, key='-INCLUDE_RASTRIGIN-')]
            ])],
            [sg.Button("Start Batch Experiments", key='-START_BATCH-')],
            [sg.Text("Progress:"), sg.ProgressBar(100, orientation='h', size=(60, 20), key='-PROGRESS-')],
            [sg.Text("Status: Ready", key='-STATUS-')],
            [sg.Multiline(size=(100, 20), key='-BATCH_OUTPUT-', disabled=True)]  # Increased size
        ]
        
        # Results analysis tab
        analysis_tab = [
            [sg.Text("Results Analysis", font=('Arial', 16, 'bold'))],
            [sg.Frame("Data Management", [
                [sg.Text("Results File:"),
                 sg.Input(key='-RESULTS_FILE-', size=(60, 1)),  # Increased size
                 sg.FileBrowse(file_types=(("CSV Files", "*.csv"),))],
                [sg.Button("Load Results", key='-LOAD_RESULTS-'),
                 sg.Button("Generate Summary", key='-GEN_SUMMARY-')]
            ])],
            [sg.Frame("Analysis Options", [
                [sg.Text("Metric to Analyze:"),
                 sg.Combo(['best_fitness', 'final_generation', 'execution_time', 'convergence_rate'],
                         default_value='best_fitness', key='-ANALYSIS_METRIC-')],
                [sg.Text("Plot Size:"),
                 sg.Combo(['Small (10x6)', 'Medium (12x8)', 'Large (15x10)', 'Extra Large (18x12)'], 
                         default_value='Large (15x10)', key='-ANALYSIS_SIZE-')],
                [sg.Button("Statistical Comparison", key='-STAT_COMPARE-'),
                 sg.Button("Plot Results", key='-PLOT_RESULTS-'),
                 sg.Button("Save Analysis Plot", key='-SAVE_ANALYSIS-')]
            ])],
            [sg.Canvas(key='-ANALYSIS_CANVAS-', size=(800, 400))],  # Smaller canvas for plots
            [sg.Multiline(size=(120, 25), key='-ANALYSIS_OUTPUT-', disabled=True)]  # Much larger text area
        ]
        
        # Main layout with tabs
        layout = [
            [sg.Text("GA Benchmark Optimization Tool", font=('Arial', 20, 'bold'),
                    justification='center', expand_x=True)],
            [sg.TabGroup([
                [sg.Tab("Function Visualization", viz_tab)],
                [sg.Tab("Single Experiment", single_exp_tab)],
                [sg.Tab("Batch Experiments", batch_exp_tab)],
                [sg.Tab("Results Analysis", analysis_tab)]
            ], expand_x=True, expand_y=True)],
            [sg.Button("Exit", key='-EXIT-')]
        ]
        
        return layout
    
    def create_window(self):
        """Create and return the main window."""
        layout = self.create_layout()
        self.window = sg.Window(
            "GA Benchmark Optimization Tool",
            layout,
            finalize=True,
            resizable=True,
            size=(1400, 900)  # Increased window size
        )
        return self.window
    
    def clear_canvas(self, canvas_key):
        """Clear the matplotlib canvas."""
        canvas_elem = self.window[canvas_key]
        for widget in canvas_elem.TKCanvas.winfo_children():
            widget.destroy()
        if self.current_canvas_agg:
            self.current_canvas_agg = None
    
    def draw_matplotlib_figure(self, canvas_elem, figure):
        """Draw matplotlib figure on PySimpleGUI canvas."""
        # Clear any existing plot first
        for widget in canvas_elem.TKCanvas.winfo_children():
            widget.destroy()
            
        figure_canvas_agg = FigureCanvasTkAgg(figure, canvas_elem.TKCanvas)
        figure_canvas_agg.draw()
        figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=1)
        self.current_canvas_agg = figure_canvas_agg
        return figure_canvas_agg
    
    def get_plot_size(self, size_string):
        """Parse plot size string and return figsize tuple."""
        size_map = {
            'Small (12x5)': (12, 5),
            'Medium (15x6)': (15, 6),
            'Large (18x8)': (18, 8),
            'Extra Large (20x10)': (20, 10),
            'Small (10x6)': (10, 6),
            'Medium (12x8)': (12, 8),
            'Large (15x10)': (15, 10),
            'Extra Large (18x12)': (18, 12)
        }
        return size_map.get(size_string, (15, 8))
    
    def visualize_function(self, values):
        """Handle function visualization."""
        func_name = values['-VIZ_FUNC-'].lower()
        resolution = int(values['-RESOLUTION-'])
        plot_size = self.get_plot_size(values['-VIZ_SIZE-'])
        
        if func_name in self.functions:
            func = self.functions[func_name]
            
            # Create figure with custom size
            fig = plt.figure(figsize=plot_size)
            
            # Create grid for evaluation
            x_range = np.linspace(func.domain[0], func.domain[1], resolution)
            y_range = np.linspace(func.domain[0], func.domain[1], resolution)
            X, Y = np.meshgrid(x_range, y_range)
            
            # Evaluate function on grid
            Z = np.zeros_like(X)
            for i in range(resolution):
                for j in range(resolution):
                    Z[i, j] = func([X[i, j], Y[i, j]])
            
            # 2D Contour plot
            ax1 = fig.add_subplot(121)
            contourf = ax1.contourf(X, Y, Z, levels=50, cmap='viridis')
            ax1.plot(func.global_optimum[0], func.global_optimum[1],
                    'r*', markersize=20, label='Global Optimum')  # Larger marker
            ax1.set_xlabel('x1', fontsize=12)
            ax1.set_ylabel('x2', fontsize=12)
            ax1.set_title(f'{func.name} - 2D Contour', fontsize=14)
            ax1.legend(fontsize=12)
            plt.colorbar(contourf, ax=ax1, label='Function Value')
            
            # 3D Surface plot
            ax2 = fig.add_subplot(122, projection='3d')
            surface = ax2.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
            ax2.scatter(func.global_optimum[0], func.global_optimum[1], func.global_minimum,
                       color='red', s=150)  # Larger scatter point
            ax2.set_xlabel('x1', fontsize=12)
            ax2.set_ylabel('x2', fontsize=12)
            ax2.set_zlabel('f(x1, x2)', fontsize=12)
            ax2.set_title(f'{func.name} - 3D Surface', fontsize=14)
            
            plt.tight_layout()
            
            # Draw on canvas
            canvas_elem = self.window['-VIZ_CANVAS-']
            self.draw_matplotlib_figure(canvas_elem, fig)
    
    def run_single_experiment(self, values):
        """Run a single GA experiment."""
        try:
            # Get configuration
            func_name = 'ackley' if values['-FUNC_ACKLEY-'] else 'rastrigin'
            representation = 'binary' if values['-REP_BINARY-'] else 'real_valued'
            
            # Validate crossover selection
            crossover = values['-CROSSOVER-']
            if representation == 'binary' and crossover not in ['one_point', 'two_point']:
                crossover = 'one_point'
            elif representation == 'real_valued' and crossover not in ['arithmetic', 'blx_alpha']:
                crossover = 'arithmetic'
            
            function = self.functions[func_name]
            
            # Create GA instance
            common_params = {
                'fitness_function': function,
                'domain': function.domain,
                'dimensions': function.dimension,
                'population_size': int(values['-POP_SIZE-']),
                'max_generations': int(values['-MAX_GEN-']),
                'crossover_rate': float(values['-CROSS_RATE-']),
                'mutation_rate': float(values['-MUT_RATE-']),
                'random_seed': 42
            }
            
            if representation == 'binary':
                ga = BinaryGA(crossover_type=crossover, **common_params)
            else:
                ga = RealValuedGA(crossover_type=crossover, **common_params)
            
            # Run experiment
            self.window['-SINGLE_OUTPUT-'].update("Running experiment...\n", append=True)
            results = ga.run()
            
            # Display results
            output = f"""
Experiment Results:
==================
Function: {func_name.title()}
Representation: {representation}
Crossover: {crossover}
Population Size: {common_params['population_size']}
Max Generations: {common_params['max_generations']}

Results:
--------
Best Fitness: {results['best_fitness']:.6f}
Best Individual: {results['best_individual']}
Generations Run: {results['generations_run']}
Fitness Evaluations: {results['fitness_evaluations']}
Execution Time: {results['execution_time']:.3f} seconds

Global Optimum: {function.global_optimum}
Global Minimum: {function.global_minimum}
Distance from Optimum: {np.linalg.norm(results['best_individual'] - function.global_optimum):.6f}
"""
            self.window['-SINGLE_OUTPUT-'].update(output, append=True)
            
        except Exception as e:
            self.window['-SINGLE_OUTPUT-'].update(f"Error: {str(e)}\n", append=True)
    
    def start_batch_experiments(self, values):
        """Start batch experiments in a separate thread."""
        if self.experiment_thread and self.experiment_thread.is_alive():
            sg.popup_error("Experiments already running!")
            return
        
        # Get configuration
        config = {
            'runs_per_config': int(values['-RUNS_PER_CONFIG-']),
            'max_fitness_evaluations': int(values['-MAX_EVALS-']),
            'include_binary': values['-INCLUDE_BINARY-'],
            'include_real': values['-INCLUDE_REAL-'],
            'include_ackley': values['-INCLUDE_ACKLEY-'],
            'include_rastrigin': values['-INCLUDE_RASTRIGIN-']
        }
        
        # Start experiment thread
        self.experiment_thread = threading.Thread(
            target=self.run_batch_experiments,
            args=(config,),
            daemon=True
        )
        self.experiment_thread.start()
        
        # Update UI
        self.window['-START_BATCH-'].update(disabled=True)
        self.window['-STATUS-'].update("Status: Running experiments...")
    
    def run_batch_experiments(self, config):
        """Run batch experiments (called in separate thread)."""
        try:
            # Create custom configurations based on user selection
            configurations = []
            functions = []
            representations = []
            
            if config['include_ackley']:
                functions.append('ackley')
            if config['include_rastrigin']:
                functions.append('rastrigin')
            
            if config['include_binary']:
                representations.append(('binary', ['one_point', 'two_point']))
            if config['include_real']:
                representations.append(('real_valued', ['arithmetic', 'blx_alpha']))
            
            # Generate configurations
            from src.experiments.config_manager import GAConfig
            for func_name in functions:
                for rep_name, crossovers in representations:
                    for crossover in crossovers:
                        configurations.append(GAConfig(
                            function_name=func_name,
                            representation=rep_name,
                            crossover_type=crossover
                        ))
            
            total_configs = len(configurations)
            total_runs = total_configs * config['runs_per_config']
            
            # Create experiment runner
            runner = ExperimentRunner(
                runs_per_config=config['runs_per_config'],
                max_fitness_evaluations=config['max_fitness_evaluations']
            )
            
            all_results = []
            completed_runs = 0
            
            for i, ga_config in enumerate(configurations):
                config_results = []
                config_name = f"{ga_config.function_name}_{ga_config.representation}_{ga_config.crossover_type}"
                
                # Update progress
                self.results_queue.put({
                    'type': 'status',
                    'message': f"Running {config_name} ({i+1}/{total_configs})"
                })
                
                for run in range(config['runs_per_config']):
                    result = runner.run_single_experiment(ga_config, run)
                    config_results.append(result)
                    completed_runs += 1
                    
                    # Update progress
                    progress = (completed_runs / total_runs) * 100
                    self.results_queue.put({
                        'type': 'progress',
                        'value': progress,
                        'message': f"Completed {completed_runs}/{total_runs} runs"
                    })
                
                all_results.extend(config_results)
            
            # Save results
            results_df = self.data_collector.create_results_dataframe(all_results)
            results_path = Path("results") / "gui_experiments.csv"
            results_df.to_csv(results_path, index=False)
            
            self.results_queue.put({
                'type': 'complete',
                'results_df': results_df,
                'results_path': str(results_path)
            })
            
        except Exception as e:
            self.results_queue.put({
                'type': 'error',
                'message': str(e)
            })
    
    def process_experiment_updates(self):
        """Process updates from experiment thread."""
        try:
            while True:
                update = self.results_queue.get_nowait()
                
                if update['type'] == 'progress':
                    self.window['-PROGRESS-'].update(current_count=int(update['value']))
                    self.window['-STATUS-'].update(f"Status: {update['message']}")
                
                elif update['type'] == 'status':
                    self.window['-STATUS-'].update(f"Status: {update['message']}")
                    self.window['-BATCH_OUTPUT-'].update(f"{update['message']}\n", append=True)
                
                elif update['type'] == 'complete':
                    self.current_results = update['results_df']
                    self.window['-PROGRESS-'].update(current_count=100)
                    self.window['-STATUS-'].update("Status: Experiments completed!")
                    self.window['-BATCH_OUTPUT-'].update(
                        f"Experiments completed! Results saved to {update['results_path']}\n",
                        append=True
                    )
                    self.window['-START_BATCH-'].update(disabled=False)
                
                elif update['type'] == 'error':
                    self.window['-STATUS-'].update(f"Status: Error - {update['message']}")
                    self.window['-BATCH_OUTPUT-'].update(f"Error: {update['message']}\n", append=True)
                    self.window['-START_BATCH-'].update(disabled=False)
                
        except queue.Empty:
            pass
    
    def load_results(self, file_path):
        """Load results from CSV file."""
        try:
            self.current_results = pd.read_csv(file_path)
            self.window['-ANALYSIS_OUTPUT-'].update(
                f"Loaded results from {file_path}\n"
                f"Shape: {self.current_results.shape}\n"
                f"Configurations: {self.current_results['config_name'].nunique()}\n",
                append=True
            )
        except Exception as e:
            sg.popup_error(f"Error loading results: {str(e)}")
    
    def generate_summary(self):
        """Generate summary statistics."""
        if self.current_results is None:
            sg.popup_error("No results loaded!")
            return
        
        try:
            summary = self.data_collector.calculate_summary_statistics(self.current_results)
            
            # Display top 5 configurations
            top_configs = summary.nsmallest(5, 'best_fitness_mean')[
                ['config_name', 'best_fitness_mean', 'best_fitness_std', 'success_mean']
            ]
            
            output = "Top 5 Configurations by Best Fitness:\n"
            output += "=" * 50 + "\n"
            output += top_configs.to_string(index=False)
            output += "\n\n"
            
            self.window['-ANALYSIS_OUTPUT-'].update(output, append=True)
            
        except Exception as e:
            sg.popup_error(f"Error generating summary: {str(e)}")
    
    def plot_results(self, metric):
        """Plot results analysis."""
        if self.current_results is None:
            sg.popup_error("No results loaded!")
            return
        
        try:
            # Get plot size
            plot_size = self.get_plot_size(self.window['-ANALYSIS_SIZE-'].get())
            
            # Create box plot with custom size
            import seaborn as sns
            fig, ax = plt.subplots(figsize=plot_size)
            
            sns.boxplot(data=self.current_results, x='config_name', y=metric, ax=ax)
            ax.set_title(f'Distribution of {metric} by Configuration', fontsize=16)
            ax.set_xlabel('Configuration', fontsize=14)
            ax.set_ylabel(metric, fontsize=14)
            ax.tick_params(axis='x', rotation=45, labelsize=12)
            ax.tick_params(axis='y', labelsize=12)
            plt.tight_layout()
            
            # Draw on canvas
            canvas_elem = self.window['-ANALYSIS_CANVAS-']
            self.draw_matplotlib_figure(canvas_elem, fig)
            
        except Exception as e:
            sg.popup_error(f"Error plotting results: {str(e)}")
    
    def plot_statistical_comparison(self, comparison_results, metric):
        """Create a small statistical comparison plot."""
        try:
            import seaborn as sns
            
            # Hardcoded small size for statistical comparison plots
            fig, ax = plt.subplots(figsize=(8, 5))
            
            # Get significant comparisons
            significant = comparison_results[comparison_results['significant'] == True]
            
            if len(significant) > 0:
                # Plot effect sizes of significant comparisons
                top_comparisons = significant.nlargest(10, 'effect_size')
                
                # Create comparison labels
                labels = [f"{row['config1'][:15]}...\nvs\n{row['config2'][:15]}..." 
                         for _, row in top_comparisons.iterrows()]
                
                bars = ax.bar(range(len(top_comparisons)), top_comparisons['effect_size'], 
                             color='steelblue', alpha=0.7)
                
                ax.set_xlabel('Configuration Comparisons', fontsize=10)
                ax.set_ylabel('Effect Size (Cohen\'s d)', fontsize=10)
                ax.set_title(f'Statistical Comparison - {metric}', fontsize=12)
                ax.set_xticks(range(len(top_comparisons)))
                ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
                
                # Add effect size interpretation lines
                ax.axhline(y=0.2, color='green', linestyle='--', alpha=0.5, label='Small effect')
                ax.axhline(y=0.5, color='orange', linestyle='--', alpha=0.5, label='Medium effect')
                ax.axhline(y=0.8, color='red', linestyle='--', alpha=0.5, label='Large effect')
                ax.legend(fontsize=8)
                
            else:
                ax.text(0.5, 0.5, 'No statistically significant\ndifferences found', 
                       ha='center', va='center', fontsize=12, transform=ax.transAxes)
                ax.set_title(f'Statistical Comparison - {metric}', fontsize=12)
            
            plt.tight_layout()
            
            # Draw on canvas
            canvas_elem = self.window['-ANALYSIS_CANVAS-']
            self.draw_matplotlib_figure(canvas_elem, fig)
            
        except Exception as e:
            sg.popup_error(f"Error creating statistical plot: {str(e)}")
    
    def run(self):
        """Main application loop."""
        self.create_window()
        
        while True:
            event, values = self.window.read(timeout=100)
            
            # Process experiment updates
            self.process_experiment_updates()
            
            if event in (sg.WIN_CLOSED, '-EXIT-'):
                break
            
            elif event == '-VISUALIZE-':
                self.visualize_function(values)
            
            elif event == '-CLEAR_VIZ-':
                self.clear_canvas('-VIZ_CANVAS-')
            
            elif event == '-SAVE_VIZ-':
                if self.current_canvas_agg:
                    filename = sg.popup_get_file('Save plot as', save_as=True, 
                                               file_types=(("PNG files", "*.png"), ("PDF files", "*.pdf")))
                    if filename:
                        self.current_canvas_agg.figure.savefig(filename, dpi=300, bbox_inches='tight')
                        sg.popup(f"Plot saved as {filename}")
                else:
                    sg.popup_error("No plot to save!")
            
            elif event == '-RUN_SINGLE-':
                self.run_single_experiment(values)
            
            elif event == '-CLEAR_SINGLE-':
                self.window['-SINGLE_OUTPUT-'].update('')
            
            elif event == '-START_BATCH-':
                self.start_batch_experiments(values)
            
            elif event == '-LOAD_RESULTS-':
                file_path = values['-RESULTS_FILE-']
                if file_path:
                    self.load_results(file_path)
            
            elif event == '-GEN_SUMMARY-':
                self.generate_summary()
            
            elif event == '-PLOT_RESULTS-':
                metric = values['-ANALYSIS_METRIC-']
                self.plot_results(metric)
            
            elif event == '-SAVE_ANALYSIS-':
                if self.current_canvas_agg:
                    filename = sg.popup_get_file('Save analysis plot as', save_as=True, 
                                               file_types=(("PNG files", "*.png"), ("PDF files", "*.pdf")))
                    if filename:
                        self.current_canvas_agg.figure.savefig(filename, dpi=300, bbox_inches='tight')
                        sg.popup(f"Analysis plot saved as {filename}")
                else:
                    sg.popup_error("No plot to save!")
            
            elif event == '-STAT_COMPARE-':
                if self.current_results is None:
                    sg.popup_error("No results loaded!")
                else:
                    # Perform statistical comparison
                    comparison = self.analyzer.analyze_configuration_differences(
                        self.current_results, values['-ANALYSIS_METRIC-']
                    )
                    
                    # Create statistical comparison plot
                    self.plot_statistical_comparison(comparison, values['-ANALYSIS_METRIC-'])
                    
                    # Show detailed text results
                    significant = comparison[comparison['significant'] == True]
                    if len(significant) > 0:
                        # Show all significant differences with detailed information
                        output = f"Statistical Comparison Results - {values['-ANALYSIS_METRIC-']}:\n"
                        output += "=" * 70 + "\n\n"
                        
                        output += f"Total comparisons: {len(comparison)}\n"
                        output += f"Significant differences: {len(significant)}\n"
                        output += f"Significance level: α = 0.05\n\n"
                        
                        output += "Significant Comparisons (sorted by effect size):\n"
                        output += "-" * 50 + "\n"
                        top_diffs = significant.nlargest(15, 'effect_size')
                        for i, (_, row) in enumerate(top_diffs.iterrows(), 1):
                            effect_magnitude = "Large" if abs(row['effect_size']) >= 0.8 else \
                                             "Medium" if abs(row['effect_size']) >= 0.5 else "Small"
                            
                            output += f"{i:2d}. {row['config1']} vs {row['config2']}\n"
                            output += f"    p-value: {row['p_value']:.6f}\n"
                            output += f"    Effect size: {row['effect_size']:.3f} ({effect_magnitude})\n"
                            output += f"    Better: {row['better_config']}\n"
                            output += f"    Config1 mean: {row['config1_mean']:.6f}\n"
                            output += f"    Config2 mean: {row['config2_mean']:.6f}\n\n"
                        
                        if len(significant) > 15:
                            output += f"... and {len(significant) - 15} more significant differences\n\n"
                        
                        # Summary by representation type
                        output += "Summary by Representation:\n"
                        output += "-" * 30 + "\n"
                        binary_configs = [c for c in self.current_results['config_name'].unique() if 'binary' in c]
                        real_configs = [c for c in self.current_results['config_name'].unique() if 'real_valued' in c]
                        
                        if binary_configs and real_configs:
                            binary_mean = self.current_results[
                                self.current_results['config_name'].isin(binary_configs)
                            ][values['-ANALYSIS_METRIC-']].mean()
                            real_mean = self.current_results[
                                self.current_results['config_name'].isin(real_configs)
                            ][values['-ANALYSIS_METRIC-']].mean()
                            
                            output += f"Binary representations mean: {binary_mean:.6f}\n"
                            output += f"Real-valued representations mean: {real_mean:.6f}\n"
                            output += f"Performance ratio: {binary_mean/real_mean:.2f}x\n\n"
                        
                    else:
                        output = f"Statistical Comparison Results - {values['-ANALYSIS_METRIC-']}:\n"
                        output += "=" * 70 + "\n\n"
                        output += "No statistically significant differences found.\n"
                        output += f"Total comparisons tested: {len(comparison)}\n"
                        output += "All p-values > 0.05\n\n"
                        output += "This could indicate:\n"
                        output += "- Similar performance across configurations\n"
                        output += "- Insufficient sample size\n"
                        output += "- High variance in results\n\n"
                    
                    self.window['-ANALYSIS_OUTPUT-'].update(output, append=True)
        
        self.window.close()


def main():
    """Main function to run the GUI application."""
    try:
        app = GAOptimizerGUI()
        app.run()
    except Exception as e:
        print(f"Error starting GUI: {e}")
        print("Please ensure PySimpleGUI is installed: pip install PySimpleGUI")


if __name__ == "__main__":
    main()
