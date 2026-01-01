import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional


def plot_metric_vs_parameter(parameter_values: np.ndarray, results: Dict[str, np.ndarray], 
                            parameter_name: str, metric_name: str, parameter_unit: str = '', 
                            metric_unit: str = '', title: str = '', show_crb: bool = False, 
                            crb_values: Optional[np.ndarray] = None, crb_label: str = 'CRB'):
    """绘制指标随参数变化的折线图。
    
    Args:
        parameter_values (np.ndarray): 参数值数组。
        results (Dict[str, np.ndarray]): 不同算法的指标结果字典，键为算法名称，值为对应指标值数组。
        parameter_name (str): 参数名称，用于x轴标签。
        metric_name (str): 指标名称，用于y轴标签和图例。
        parameter_unit (str, optional): 参数单位，用于x轴标签。默认值为空字符串。
        metric_unit (str, optional): 指标单位，用于y轴标签。默认值为空字符串。
        title (str, optional): 图标题。默认值为空字符串。
        show_crb (bool, optional): 是否显示CRB曲线。默认值为False。
        crb_values (Optional[np.ndarray], optional): CRB值数组。默认值为None。
        crb_label (str, optional): CRB曲线的图例标签。默认值为'CRB'。
    """
    plt.figure(figsize=(12, 6))
    
    # 设置x轴标签
    xlabel = f'{parameter_name}'
    if parameter_unit:
        xlabel += f' ({parameter_unit})'
    
    # 设置y轴标签
    ylabel = f'{metric_name}'
    if metric_unit:
        ylabel += f' ({metric_unit})'
    
    # 定义颜色循环
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange', 'purple', 'brown']
    markers = ['x', 'o', 's', '^', 'v', '<', '>', 'D', 'p', '*']
    
    # 绘制CRB曲线（如果需要）
    if show_crb and crb_values is not None:
        plt.semilogy(parameter_values, crb_values, '--k', linewidth=2, label=crb_label)
    
    # 绘制每个算法的曲线
    for i, (algorithm, metric_values) in enumerate(results.items()):
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]
        plt.semilogy(parameter_values, metric_values, f'-{marker}', color=color, 
                    linewidth=1.5, markersize=8, label=algorithm)
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.legend(loc='lower left', ncol=2, fontsize=10)
    
    if title:
        plt.title(title)
    else:
        plt.title(f'{metric_name} vs. {parameter_name} for Multiple DOA Algorithms')
    
    plt.margins(x=0)
    plt.tight_layout()
    plt.show()


def plot_scatter_estimates(true_angles: np.ndarray, estimates: np.ndarray, 
                          algorithm_name: str, angle_unit: str = 'rad'):
    """绘制真实角度与估计角度的散点图。
    
    Args:
        true_angles (np.ndarray): 真实角度数组，形状为(n_monte_carlo, n_sources)或(n_sources,)。
        estimates (np.ndarray): 估计角度数组，形状为(n_monte_carlo, n_sources)。
        algorithm_name (str): 算法名称，用于标题。
        angle_unit (str, optional): 角度单位，'rad'或'deg'。默认值为'rad'。
    """
    # 确保true_angles形状与estimates一致
    if true_angles.ndim == 1:
        true_angles = np.tile(true_angles, (estimates.shape[0], 1))
    
    # 转换为度（如果需要）
    if angle_unit == 'deg':
        true_angles = np.rad2deg(true_angles)
        estimates = np.rad2deg(estimates)
        angle_label = 'Angle (degrees)'
    else:
        angle_label = 'Angle (radians)'
    
    n_sources = true_angles.shape[1]
    
    plt.figure(figsize=(12, 6))
    
    # 定义颜色循环
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange', 'purple', 'brown']
    markers = ['x', 'o', 's', '^', 'v', '<', '>', 'D', 'p', '*']
    
    # 绘制每个信源的散点图
    for i in range(n_sources):
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]
        plt.scatter(true_angles[:, i], estimates[:, i], color=color, marker=marker, 
                   s=50, alpha=0.6, label=f'Source {i+1}')
    
    # 绘制理想线（y = x）
    min_val = min(np.min(true_angles), np.min(estimates))
    max_val = max(np.max(true_angles), np.max(estimates))
    plt.plot([min_val, max_val], [min_val, max_val], '--k', linewidth=2, label='Ideal')
    
    plt.xlabel(f'True {angle_label}')
    plt.ylabel(f'Estimated {angle_label}')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='upper left', ncol=2, fontsize=10)
    plt.title(f'True vs. Estimated Angles for {algorithm_name}')
    
    plt.axis('equal')
    plt.tight_layout()
    plt.show()


def plot_cdf(estimates: np.ndarray, true_angles: np.ndarray, algorithm_name: str, 
             metric_name: str = 'Error', metric_unit: str = 'rad'):
    """绘制估计误差的CDF（累积分布函数）图。
    
    Args:
        estimates (np.ndarray): 估计角度数组，形状为(n_monte_carlo, n_sources)。
        true_angles (np.ndarray): 真实角度数组，形状为(n_monte_carlo, n_sources)或(n_sources,)。
        algorithm_name (str): 算法名称，用于标题和图例。
        metric_name (str, optional): 误差指标名称，用于x轴标签。默认值为'Error'。
        metric_unit (str, optional): 误差单位，用于x轴标签。默认值为'rad'。
    """
    # 确保true_angles形状与estimates一致
    if true_angles.ndim == 1:
        true_angles = np.tile(true_angles, (estimates.shape[0], 1))
    
    # 计算误差
    errors = np.abs(estimates - true_angles)
    # 转换为度（如果需要）
    if metric_unit == 'deg':
        errors = np.rad2deg(errors)
    
    n_sources = errors.shape[1]
    
    plt.figure(figsize=(12, 6))
    
    # 定义颜色循环
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange', 'purple', 'brown']
    
    # 绘制每个信源的CDF
    for i in range(n_sources):
        color = colors[i % len(colors)]
        # 提取第i个信源的误差
        source_errors = errors[:, i]
        # 排序误差
        sorted_errors = np.sort(source_errors)
        # 计算CDF值
        cdf = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
        # 绘制CDF
        plt.plot(sorted_errors, cdf, '-', color=color, linewidth=2, 
                label=f'{algorithm_name} - Source {i+1}')
    
    plt.xlabel(f'{metric_name} ({metric_unit})')
    plt.ylabel('CDF')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='lower right', fontsize=10)
    plt.title(f'CDF of {metric_name} for {algorithm_name}')
    
    plt.margins(x=0)
    plt.tight_layout()
    plt.show()


def plot_histogram(estimates: np.ndarray, true_angles: np.ndarray, algorithm_name: str, 
                   bins: int = 50, metric_unit: str = 'rad'):
    """绘制估计误差的直方图。
    
    Args:
        estimates (np.ndarray): 估计角度数组，形状为(n_monte_carlo, n_sources)。
        true_angles (np.ndarray): 真实角度数组，形状为(n_monte_carlo, n_sources)或(n_sources,)。
        algorithm_name (str): 算法名称，用于标题。
        bins (int, optional): 直方图的分箱数。默认值为50。
        metric_unit (str, optional): 角度单位，'rad'或'deg'。默认值为'rad'。
    """
    # 确保true_angles形状与estimates一致
    if true_angles.ndim == 1:
        true_angles = np.tile(true_angles, (estimates.shape[0], 1))
    
    # 计算误差
    errors = estimates - true_angles
    # 转换为度（如果需要）
    if metric_unit == 'deg':
        errors = np.rad2deg(errors)
        angle_label = 'Error (degrees)'
    else:
        angle_label = 'Error (radians)'
    
    n_sources = errors.shape[1]
    
    # 创建子图
    fig, axes = plt.subplots(n_sources, 1, figsize=(12, 3 * n_sources))
    if n_sources == 1:
        axes = [axes]  # 确保axes是列表
    
    # 定义颜色循环
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange', 'purple', 'brown']
    
    # 为每个信源绘制直方图
    for i, ax in enumerate(axes):
        color = colors[i % len(colors)]
        # 提取第i个信源的误差
        source_errors = errors[:, i]
        # 绘制直方图
        ax.hist(source_errors, bins=bins, color=color, alpha=0.7, density=True)
        # 绘制真实角度位置的垂直线
        true_angle = true_angles[0, i]  # 所有蒙特卡洛模拟中真实角度相同
        ax.axvline(x=0, color='k', linestyle='--', linewidth=2, label='Zero Error')
        
        ax.set_xlabel(angle_label)
        ax.set_ylabel('Probability Density')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(loc='upper right', fontsize=10)
        ax.set_title(f'Error Distribution for {algorithm_name} - Source {i+1}')
    
    plt.tight_layout()
    plt.show()


def plot_resolution_comparison(delta_thetas: np.ndarray, success_rates: Dict[str, np.ndarray], 
                               parameter_unit: str = 'rad', title: str = ''):
    """绘制不同算法的分辨率比较图。
    
    Args:
        delta_thetas (np.ndarray): 角度间隔数组。
        success_rates (Dict[str, np.ndarray]): 不同算法的成功分辨率字典，键为算法名称，值为成功分辨率数组。
        parameter_unit (str, optional): 角度单位，'rad'或'deg'。默认值为'rad'。
        title (str, optional): 图标题。默认值为空字符串。
    """
    plt.figure(figsize=(12, 6))
    
    # 设置x轴标签
    if parameter_unit == 'deg':
        xlabel = 'Angular Separation (degrees)'
    else:
        xlabel = 'Angular Separation (radians)'
    
    # 定义颜色循环
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange', 'purple', 'brown']
    markers = ['x', 'o', 's', '^', 'v', '<', '>', 'D', 'p', '*']
    
    # 绘制每个算法的曲线
    for i, (algorithm, rates) in enumerate(success_rates.items()):
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]
        plt.plot(delta_thetas, rates, f'-{marker}', color=color, 
                linewidth=1.5, markersize=8, label=algorithm)
    
    # 绘制50%成功率线
    plt.axhline(y=0.5, color='k', linestyle='--', linewidth=2, label='50% Success Rate')
    
    plt.xlabel(xlabel)
    plt.ylabel('Success Rate')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='lower right', ncol=2, fontsize=10)
    
    if title:
        plt.title(title)
    else:
        plt.title('Resolution Comparison for Multiple DOA Algorithms')
    
    plt.ylim([0, 1.05])
    plt.margins(x=0)
    plt.tight_layout()
    plt.show()
