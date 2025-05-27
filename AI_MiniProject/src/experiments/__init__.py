"""
Experiments package for running GA optimization experiments.

This package provides tools for running systematic experiments
with different GA configurations and collecting results.
"""

from .experiment_runner import ExperimentRunner
from .config_manager import ConfigManager
from .data_collector import DataCollector

__all__ = ['ExperimentRunner', 'ConfigManager', 'DataCollector']
