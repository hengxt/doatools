import numpy as np
import matplotlib.pyplot as plt
import unittest
from doatools.plotting.plot_performance import (
    get_global_config,
    set_doa_method_style,
    get_doa_method_style,
    plot_metric_vs_parameter,
    plot_scatter_estimates,
    plot_cdf
)

class TestPlotPerformance(unittest.TestCase):
    
    def test_get_global_config(self):
        """Test if get_global_config returns the correct configuration dictionary."""
        config = get_global_config()
        assert isinstance(config, dict)
        assert 'doa_method_styles' in config
        assert 'default_colors' in config
        assert 'default_markers' in config
    
    def test_set_doa_method_style(self):
        """Test if set_doa_method_style correctly updates the method style."""
        # Save original config
        original_config = get_global_config().copy()
        
        # Test setting style for a known method
        set_doa_method_style('MUSIC', color='red', marker='*')
        assert get_global_config()['doa_method_styles']['MUSIC']['color'] == 'red'
        assert get_global_config()['doa_method_styles']['MUSIC']['marker'] == '*'
        
        # Test setting style for a new method
        set_doa_method_style('NewMethod', color='green', marker='o')
        assert 'NewMethod' in get_global_config()['doa_method_styles']
        assert get_global_config()['doa_method_styles']['NewMethod']['color'] == 'green'
        
        # Restore original config
        get_global_config().update(original_config)
    
    def test_get_doa_method_style(self):
        """Test if get_doa_method_style returns the correct color and marker."""
        # Test for known method
        color, marker = get_doa_method_style('MUSIC')
        assert isinstance(color, str)
        assert isinstance(marker, str)
        
        # Test for unknown method
        color, marker = get_doa_method_style('UnknownMethod')
        assert isinstance(color, str)
        assert isinstance(marker, str)
        
        # Test with index
        color1, marker1 = get_doa_method_style('UnknownMethod', 0)
        color2, marker2 = get_doa_method_style('UnknownMethod', 1)
        assert (color1, marker1) != (color2, marker2)
    
    def test_plot_metric_vs_parameter(self):
        """Test if plot_metric_vs_parameter runs without errors."""
        # Create sample data
        parameter_values = np.linspace(0, 10, 5)
        results = {
            'MUSIC': np.ones(5),
            'RootMUSIC': np.ones(5) * 0.5
        }
        
        # Test with default parameters
        fig, ax = plt.subplots()
        plot_metric_vs_parameter(parameter_values, results, 'SNR', 'RMSE', ax=ax)
        plt.close(fig)
        
        # Test with CRB
        fig, ax = plt.subplots()
        plot_metric_vs_parameter(parameter_values, results, 'SNR', 'RMSE', 
                                show_crb=True, crb_values=np.ones(5) * 0.1, ax=ax)
        plt.close(fig)
    
    def test_plot_scatter_estimates(self):
        """Test if plot_scatter_estimates runs without errors."""
        # Create sample data
        n_monte_carlo = 10
        n_sources = 2
        true_angles = np.tile(np.array([0.1, 0.2]), (n_monte_carlo, 1))
        estimates = true_angles + np.random.normal(0, 0.01, true_angles.shape)
        
        # Test with default parameters
        fig, ax = plt.subplots()
        plot_scatter_estimates(true_angles, estimates, 'MUSIC', ax=ax)
        plt.close(fig)
        
        # Test with degrees
        fig, ax = plt.subplots()
        plot_scatter_estimates(true_angles, estimates, 'MUSIC', angle_unit='deg', ax=ax)
        plt.close(fig)
    
    def test_plot_cdf(self):
        """Test if plot_cdf runs without errors."""
        # Create sample data for single algorithm
        n_monte_carlo = 10
        n_sources = 2
        true_angles = np.tile(np.array([0.1, 0.2]), (n_monte_carlo, 1))
        estimates = true_angles + np.random.normal(0, 0.01, true_angles.shape)
        
        # Test single algorithm mode
        fig, ax = plt.subplots()
        plot_cdf(estimates, true_angles, 'MUSIC', ax=ax)
        plt.close(fig)
        
        # Test multi-algorithm mode
        results = {
            'MUSIC': estimates,
            'RootMUSIC': true_angles + np.random.normal(0, 0.005, true_angles.shape)
        }
        fig, ax = plt.subplots()
        plot_cdf(results, true_angles, ax=ax)
        plt.close(fig)
        
        # Test different metric types
        fig, ax = plt.subplots()
        plot_cdf(estimates, true_angles, 'MUSIC', metric_type='rms', ax=ax)
        plt.close(fig)
        
        fig, ax = plt.subplots()
        plot_cdf(estimates, true_angles, 'MUSIC', metric_type='mae', ax=ax)
        plt.close(fig)
