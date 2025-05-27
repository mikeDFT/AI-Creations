"""
GUI Application Launcher for GA Benchmark Optimization.

This script launches the PySimpleGUI-based interface for running
genetic algorithm optimization experiments.
"""

import sys
from pathlib import Path

# Add src directory to path for imports
src_path = Path(__file__).parent
sys.path.insert(0, str(src_path))

from gui.main_window import GAOptimizerGUI


def main():
    """Launch the GUI application."""
    print("Starting GA Benchmark Optimization GUI...")
    print("=" * 50)
    
    try:
        app = GAOptimizerGUI()
        app.run()
    except Exception as e:
        print(f"Error starting GUI: {e}")
        input("Press Enter to exit...")


if __name__ == "__main__":
    main()
