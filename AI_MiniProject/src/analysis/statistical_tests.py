"""
Statistical analysis tools for GA experiment results.

This module provides functions for conducting statistical tests
to determine significance of performance differences between GA configurations.
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Any
import itertools


class StatisticalAnalyzer:
    """Performs statistical analysis on GA experimental results."""
    
    def __init__(self, alpha: float = 0.05):
        """
        Initialize statistical analyzer.
        
        Args:
            alpha: Significance level for statistical tests
        """
        self.alpha = alpha
    
    def t_test_two_groups(self, group1: np.ndarray, group2: np.ndarray) -> Dict[str, Any]:
        """
        Perform independent t-test between two groups.
        
        Args:
            group1, group2: Arrays of values to compare
            
        Returns:
            Dictionary with test results
        """
        # Check normality (Shapiro-Wilk test)
        norm1 = stats.shapiro(group1)[1] > self.alpha if len(group1) >= 3 else True
        norm2 = stats.shapiro(group2)[1] > self.alpha if len(group2) >= 3 else True
        
        if norm1 and norm2:
            # Use t-test if both groups are normal
            statistic, p_value = stats.ttest_ind(group1, group2, equal_var=False)
            test_used = "Welch's t-test"
        else:
            # Use Mann-Whitney U test if normality assumption is violated
            statistic, p_value = stats.mannwhitneyu(group1, group2, alternative='two-sided')
            test_used = "Mann-Whitney U test"
        
        effect_size = self.calculate_effect_size(group1, group2)
        
        return {
            'test_used': test_used,
            'statistic': statistic,
            'p_value': p_value,
            'significant': p_value < self.alpha,
            'effect_size': effect_size,
            'group1_mean': np.mean(group1),
            'group2_mean': np.mean(group2),
            'group1_std': np.std(group1, ddof=1),
            'group2_std': np.std(group2, ddof=1)
        }
    
    def calculate_effect_size(self, group1: np.ndarray, group2: np.ndarray) -> float:
        """
        Calculate Cohen's d effect size.
        
        Args:
            group1, group2: Arrays of values
            
        Returns:
            Effect size (Cohen's d)
        """
        n1, n2 = len(group1), len(group2)
        pooled_std = np.sqrt(((n1 - 1) * np.var(group1, ddof=1) + 
                             (n2 - 1) * np.var(group2, ddof=1)) / (n1 + n2 - 2))
        
        if pooled_std == 0:
            return 0.0
        
        return (np.mean(group1) - np.mean(group2)) / pooled_std
    
    def wilcoxon_signed_rank_test(self, before: np.ndarray, after: np.ndarray) -> Dict[str, Any]:
        """
        Perform Wilcoxon signed-rank test for paired samples.
        
        Args:
            before, after: Paired arrays of values
            
        Returns:
            Dictionary with test results
        """
        statistic, p_value = stats.wilcoxon(before, after, alternative='two-sided')
        
        return {
            'test_used': 'Wilcoxon signed-rank test',
            'statistic': statistic,
            'p_value': p_value,
            'significant': p_value < self.alpha,
            'median_difference': np.median(after - before)
        }
    
    def anova_multiple_groups(self, groups: List[np.ndarray], group_names: List[str]) -> Dict[str, Any]:
        """
        Perform one-way ANOVA for multiple groups.
        
        Args:
            groups: List of arrays for each group
            group_names: Names for each group
            
        Returns:
            Dictionary with ANOVA results
        """
        # Check if all groups have the same variance (Levene's test)
        levene_stat, levene_p = stats.levene(*groups)
        equal_variances = levene_p > self.alpha
        
        if equal_variances:
            # Standard ANOVA
            f_stat, p_value = stats.f_oneway(*groups)
            test_used = "One-way ANOVA"
        else:
            # Welch's ANOVA
            f_stat, p_value = stats.f_oneway(*groups)  # scipy doesn't have Welch's, use Kruskal-Wallis instead
            f_stat, p_value = stats.kruskal(*groups)
            test_used = "Kruskal-Wallis test"
        
        result = {
            'test_used': test_used,
            'statistic': f_stat,
            'p_value': p_value,
            'significant': p_value < self.alpha,
            'levene_statistic': levene_stat,
            'levene_p_value': levene_p,
            'equal_variances': equal_variances
        }
        
        # Post-hoc tests if significant
        if p_value < self.alpha and len(groups) > 2:
            result['posthoc'] = self.posthoc_pairwise_tests(groups, group_names)
        
        return result
    
    def posthoc_pairwise_tests(self, groups: List[np.ndarray], group_names: List[str]) -> Dict[str, Any]:
        """
        Perform pairwise post-hoc tests with Bonferroni correction.
        
        Args:
            groups: List of arrays for each group
            group_names: Names for each group
            
        Returns:
            Dictionary with pairwise test results
        """
        pairwise_results = {}
        n_comparisons = len(groups) * (len(groups) - 1) // 2
        corrected_alpha = self.alpha / n_comparisons  # Bonferroni correction
        
        for i, j in itertools.combinations(range(len(groups)), 2):
            group1, group2 = groups[i], groups[j]
            name1, name2 = group_names[i], group_names[j]
            
            test_result = self.t_test_two_groups(group1, group2)
            test_result['corrected_alpha'] = corrected_alpha
            test_result['significant_corrected'] = test_result['p_value'] < corrected_alpha
            
            pairwise_results[f"{name1}_vs_{name2}"] = test_result
        
        return pairwise_results
    
    def analyze_configuration_differences(self, results_df: pd.DataFrame, 
                                        metric: str = 'best_fitness') -> pd.DataFrame:
        """
        Analyze differences between all GA configurations.
        
        Args:
            results_df: DataFrame with experimental results
            metric: Metric to analyze
            
        Returns:
            DataFrame with statistical test results
        """
        configurations = results_df['config_name'].unique()
        comparison_results = []
        
        for config1, config2 in itertools.combinations(configurations, 2):
            group1 = results_df[results_df['config_name'] == config1][metric].values
            group2 = results_df[results_df['config_name'] == config2][metric].values
            
            test_result = self.t_test_two_groups(group1, group2)
            
            comparison_results.append({
                'config1': config1,
                'config2': config2,
                'test_used': test_result['test_used'],
                'p_value': test_result['p_value'],
                'significant': test_result['significant'],
                'effect_size': test_result['effect_size'],
                'config1_mean': test_result['group1_mean'],
                'config2_mean': test_result['group2_mean'],
                'better_config': config1 if test_result['group1_mean'] < test_result['group2_mean'] else config2
            })
        
        return pd.DataFrame(comparison_results)
