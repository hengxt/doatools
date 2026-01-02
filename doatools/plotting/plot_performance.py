import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union

# Global configuration that allows user customization
_GLOBAL_CONFIG = {
    # Mapping between DOA estimation methods and their colors/markers
    'doa_method_styles': {
        # Subspace methods
        'RootMUSIC': {'color': 'b', 'marker': 'x'},
        'MUSIC': {'color': 'g', 'marker': 'o'},
        'ESPRIT': {'color': 'r', 'marker': 's'},
        'MinNorm': {'color': 'purple', 'marker': '*'},
        # Beamforming methods
        'MVDR': {'color': 'c', 'marker': '^'},
        'Bartlett': {'color': 'm', 'marker': 'v'},
        # Other methods
        'Interferometer': {'color': 'y', 'marker': '<'},
        'CoarrayACMBuilder': {'color': 'orange', 'marker': '>'}
    },
    # Default color cycle for unknown methods
    'default_colors': ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange', 'purple', 'brown'],
    # Default marker cycle for unknown methods
    'default_markers': ['x', 'o', 's', '^', 'v', '<', '>', 'D', 'p', '*']
}

# Set global font and legend configuration
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.sans-serif': ['Times New Roman', 'Arial', 'DejaVu Sans'],
})


def get_global_config():
    """Gets the global configuration dictionary.
    
    Returns:
        dict: The global configuration dictionary, which can be modified directly to globally change plotting styles.
    """
    return _GLOBAL_CONFIG


def set_doa_method_style(method_name, color=None, marker=None):
    """Sets the style for a specific DOA estimation method.
    
    Args:
        method_name (str): Name of the DOA estimation method.
        color (str, optional): Color string, e.g., 'b', 'g', 'r'.
        marker (str, optional): Marker string, e.g., 'x', 'o', 's'.
    """
    if method_name not in _GLOBAL_CONFIG['doa_method_styles']:
        _GLOBAL_CONFIG['doa_method_styles'][method_name] = {}
    if color is not None:
        _GLOBAL_CONFIG['doa_method_styles'][method_name]['color'] = color
    if marker is not None:
        _GLOBAL_CONFIG['doa_method_styles'][method_name]['marker'] = marker


def get_doa_method_style(method_name, index=0):
    """Gets the style for a specific DOA estimation method.
    
    Args:
        method_name (str): Name of the DOA estimation method.
        index (int, optional): Index used to get style from default lists when the method name is not in the configuration.
            Default value is 0.
    
    Returns:
        tuple: (color, marker) tuple.
    """
    # Check if it's a known method
    config = _GLOBAL_CONFIG['doa_method_styles']
    default_colors = _GLOBAL_CONFIG['default_colors']
    default_markers = _GLOBAL_CONFIG['default_markers']
    
    if method_name in config:
        style = config[method_name]
        color = style.get('color', default_colors[index % len(default_colors)])
        marker = style.get('marker', default_markers[index % len(default_markers)])
        return (color, marker)
    else:
        # Unknown method, use default style
        return (
            default_colors[index % len(default_colors)],
            default_markers[index % len(default_markers)]
        )


def plot_metric_vs_parameter(parameter_values: np.ndarray, results: Dict[str, np.ndarray], 
                            parameter_name: str, metric_name: str, parameter_unit: str = '', 
                            metric_unit: str = '', show_crb: bool = False, 
                            crb_values: Optional[np.ndarray] = None, crb_label: str = 'CRB',
                            ax: Optional[plt.Axes] = None):
    """Plots a line graph of metric values versus parameter values.
    
    Args:
        parameter_values (np.ndarray): Array of parameter values.
        results (Dict[str, np.ndarray]): Dictionary of metric results for different algorithms, 
            where keys are algorithm names and values are corresponding metric value arrays.
        parameter_name (str): Parameter name, used for the x-axis label.
        metric_name (str): Metric name, used for the y-axis label and legend.
        parameter_unit (str, optional): Parameter unit, used for the x-axis label. Default is an empty string.
        metric_unit (str, optional): Metric unit, used for the y-axis label. Default is an empty string.
        show_crb (bool, optional): Whether to show the CRB curve. Default value is False.
        crb_values (Optional[np.ndarray], optional): Array of CRB values. Default value is None.
        crb_label (str, optional): Legend label for the CRB curve. Default value is 'CRB'.
        ax (Optional[plt.Axes], optional): Externally provided matplotlib axes object. If None, a new figure will be created.
            Default value is None.
    """
    # Create or use the provided axes
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
        show_plot = True
    else:
        show_plot = False
    
    # Set x-axis label
    xlabel = f'{parameter_name}'
    if parameter_unit:
        xlabel += f' ({parameter_unit})'
    
    # Set y-axis label
    ylabel = f'{metric_name}'
    if metric_unit:
        ylabel += f' ({metric_unit})'
    
    # Plot CRB curve if needed
    if show_crb and crb_values is not None:
        ax.semilogy(parameter_values, crb_values, '--k', linewidth=2, label=crb_label)
    
    # Plot each algorithm's curve, using colors and markers corresponding to DOA methods
    for i, (algorithm, metric_values) in enumerate(results.items()):
        # Extract method name from algorithm name (remove trailing numbers)
        method_name = algorithm
        # Handle cases like 'RootMUSIC1D'
        if method_name.endswith('1D'):
            method_name = method_name[:-2]
        # Handle cases like 'CoarrayACMBuilder1D'
        if method_name.endswith('Builder'):
            method_name = method_name[:-7]
        
        # Get corresponding color and marker
        color, marker = get_doa_method_style(method_name, i)
        ax.semilogy(parameter_values, metric_values, f'-{marker}', color=color, 
                   linewidth=1.5, markersize=8, label=algorithm)
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, which='both', linestyle='--', alpha=0.7)
    ax.legend(ncol=2)
    
    ax.margins(x=0)
    
    if show_plot:
        plt.tight_layout()
        plt.show()


def plot_scatter_estimates(true_angles: np.ndarray, estimates: np.ndarray, 
                          algorithm_name: str, angle_unit: str = 'rad',
                          ax: Optional[plt.Axes] = None):
    """Plots a scatter plot of true angles versus estimated angles.
    
    Args:
        true_angles (np.ndarray): Array of true angles, with shape (n_monte_carlo, n_sources) or (n_sources,).
        estimates (np.ndarray): Array of estimated angles, with shape (n_monte_carlo, n_sources).
        algorithm_name (str): Algorithm name, used for the title.
        angle_unit (str, optional): Angle unit, 'rad' or 'deg'. Default value is 'rad'.
        ax (Optional[plt.Axes], optional): Externally provided matplotlib axes object. If None, a new figure will be created.
            Default value is None.
    """
    # Ensure true_angles has the same shape as estimates
    if true_angles.ndim == 1:
        true_angles = np.tile(true_angles, (estimates.shape[0], 1))
    
    # Convert to degrees if needed
    if angle_unit == 'deg':
        true_angles = np.rad2deg(true_angles)
        estimates = np.rad2deg(estimates)
        angle_label = 'Angle (degrees)'
    else:
        angle_label = 'Angle (radians)'
    
    n_sources = true_angles.shape[1]
    
    # Create or use the provided axes
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
        show_plot = True
    else:
        show_plot = False
    
    # Plot scatter points for each source
    for i in range(n_sources):
        # For scatter plots, we use default styles corresponding to source indices
        # since this is for different sources of a single algorithm
        color, marker = get_doa_method_style(algorithm_name, i)
        ax.scatter(true_angles[:, i], estimates[:, i], color=color, marker=marker, 
                   s=50, alpha=0.6, label=f'Source {i+1}')
    
    # Plot the ideal line (y = x)
    min_val = min(np.min(true_angles), np.min(estimates))
    max_val = max(np.max(true_angles), np.max(estimates))
    ax.plot([min_val, max_val], [min_val, max_val], '--k', linewidth=2, label='Ideal')
    
    ax.set_xlabel(f'True {angle_label}')
    ax.set_ylabel(f'Estimated {angle_label}')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(ncol=2)
    
    ax.axis('equal')
    
    if show_plot:
        plt.tight_layout()
        plt.show()


def plot_cdf(estimates: Union[np.ndarray, Dict[str, np.ndarray]], true_angles: np.ndarray, 
             algorithm_name: Union[str, None] = None, metric_name: str = 'Error', 
             metric_unit: str = 'rad', metric_type: str = 'absolute',
             ax: Optional[plt.Axes] = None):
    """Plots the CDF (Cumulative Distribution Function) of estimation errors.
    
    Supports two modes:
    1. Single algorithm mode: Plots the CDF for a single algorithm
    2. Multi-algorithm comparison mode: Plots CDF comparisons for multiple algorithms
    
    Args:
        estimates (Union[np.ndarray, Dict[str, np.ndarray]]): 
            - Single algorithm: Array of estimated angles with shape (n_monte_carlo, n_sources).
            - Multi-algorithm comparison: Dictionary where keys are algorithm names and values are corresponding estimated angle arrays.
        true_angles (np.ndarray): Array of true angles with shape (n_monte_carlo, n_sources) or (n_sources,).
        algorithm_name (Union[str, None], optional): 
            - Single algorithm: Algorithm name, used for the title and legend.
            - Multi-algorithm comparison: None, this parameter is ignored.
            Default value is None.
        metric_name (str, optional): Error metric name, used for the x-axis label. Default value is 'Error'.
        metric_unit (str, optional): Error unit, used for the x-axis label. Default value is 'rad'.
        metric_type (str, optional): Error type, with options:
            - 'absolute': Absolute error
            - 'rms': Root Mean Square error
            - 'mae': Mean Absolute Error
            Default value is 'absolute'.
        ax (Optional[plt.Axes], optional): Externally provided matplotlib axes object. If None, a new figure will be created.
            Default value is None.
    """
    # Create or use the provided axes
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
        show_plot = True
    else:
        show_plot = False
    
    # Ensure true_angles has the correct shape
    if true_angles.ndim == 1:
        # Single algorithm case
        if isinstance(estimates, np.ndarray):
            true_angles_repeated = np.tile(true_angles, (estimates.shape[0], 1))
        # Multi-algorithm case, get shape from first algorithm
        else:
            first_alg = next(iter(estimates.keys()))
            true_angles_repeated = np.tile(true_angles, (estimates[first_alg].shape[0], 1))
    else:
        true_angles_repeated = true_angles
    
    # Handle single algorithm case
    if isinstance(estimates, np.ndarray):
        if algorithm_name is None:
            algorithm_name = 'Algorithm'
        
        # Calculate errors
        if metric_type == 'absolute':
            errors = np.abs(estimates - true_angles_repeated)
        elif metric_type == 'rms':
            errors = np.sqrt(np.mean(np.square(estimates - true_angles_repeated), axis=1))
            errors = errors[:, np.newaxis]  # Convert to (n_monte_carlo, 1)
        elif metric_type == 'mae':
            errors = np.mean(np.abs(estimates - true_angles_repeated), axis=1)
            errors = errors[:, np.newaxis]  # Convert to (n_monte_carlo, 1)
        else:
            raise ValueError(f"Unknown metric_type: {metric_type}")
        
        # Convert to degrees if needed
        if metric_unit == 'deg':
            if metric_type in ['rms', 'mae']:
                errors = np.rad2deg(errors)
            else:
                errors = np.rad2deg(errors)
        
        n_sources = errors.shape[1]
        
        # Plot CDF for each source
        for i in range(n_sources):
            # Get corresponding color and marker
            color, _ = get_doa_method_style(algorithm_name, i)
            # Extract errors for the i-th source
            source_errors = errors[:, i]
            # Sort errors
            sorted_errors = np.sort(source_errors)
            # Calculate CDF values
            cdf = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
            # Plot CDF
            ax.plot(sorted_errors, cdf, '-', color=color, linewidth=2, 
                    label=f'{algorithm_name} - Source {i+1}')
    # Handle multi-algorithm comparison case
    elif isinstance(estimates, dict):
        # Iterate through each algorithm
        for i, (alg_name, alg_estimates) in enumerate(estimates.items()):
            # Ensure true angles have the same shape as current algorithm's estimates
            if true_angles.ndim == 1:
                current_true_angles = np.tile(true_angles, (alg_estimates.shape[0], 1))
            else:
                current_true_angles = true_angles
            
            # Calculate errors
            if metric_type == 'absolute':
                errors = np.abs(alg_estimates - current_true_angles)
                # Handle each source individually
                n_sources = errors.shape[1]
                for j in range(n_sources):
                    source_errors = errors[:, j]
                    sorted_errors = np.sort(source_errors)
                    cdf = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
                    color, _ = get_doa_method_style(alg_name, j)
                    ax.plot(sorted_errors, cdf, '-', color=color, linewidth=2, 
                            label=f'{alg_name} - Source {j+1}')
            elif metric_type in ['rms', 'mae']:
                # Calculate overall metric for each Monte Carlo sample
                if metric_type == 'rms':
                    # RMS for each sample (root mean square of all sources)
                    sample_errors = np.sqrt(np.mean(np.square(alg_estimates - current_true_angles), axis=1))
                else:  # mae
                    # MAE for each sample (mean absolute error of all sources)
                    sample_errors = np.mean(np.abs(alg_estimates - current_true_angles), axis=1)
                
                # Convert to degrees if needed
                if metric_unit == 'deg':
                    sample_errors = np.rad2deg(sample_errors)
                
                # Sort errors
                sorted_errors = np.sort(sample_errors)
                # Calculate CDF values
                cdf = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
                # Get corresponding color and marker
                color, marker = get_doa_method_style(alg_name, i)
                # Plot CDF
                ax.plot(sorted_errors, cdf, '-', color=color, linewidth=2, 
                        label=alg_name)
            else:
                raise ValueError(f"Unknown metric_type: {metric_type}")
    else:
        raise ValueError(f"estimates must be either np.ndarray or dict, got {type(estimates)}")
    
    ax.set_xlabel(f'{metric_name} ({metric_unit})')
    ax.set_ylabel('CDF')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()
    
    ax.margins(x=0)
    
    if show_plot:
        plt.tight_layout()
        plt.show()

