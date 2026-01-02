import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union

# 全局配置，允许用户自定义
_GLOBAL_CONFIG = {
    # DOA估计方法与颜色、标记的对应关系
    'doa_method_styles': {
        # 子空间方法
        'RootMUSIC': {'color': 'b', 'marker': 'x'},
        'MUSIC': {'color': 'g', 'marker': 'o'},
        'ESPRIT': {'color': 'r', 'marker': 's'},
        'MinNorm': {'color': 'purple', 'marker': '*'},
        # 波束形成方法
        'MVDR': {'color': 'c', 'marker': '^'},
        'Bartlett': {'color': 'm', 'marker': 'v'},
        # 其他方法
        'Interferometer': {'color': 'y', 'marker': '<'},
        'CoarrayACMBuilder': {'color': 'orange', 'marker': '>'}
    },
    # 默认颜色循环，当遇到未知方法时使用
    'default_colors': ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange', 'purple', 'brown'],
    # 默认标记循环
    'default_markers': ['x', 'o', 's', '^', 'v', '<', '>', 'D', 'p', '*']
}

# 设置全局字体和图例配置
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.sans-serif': ['Times New Roman', 'Arial', 'DejaVu Sans'],
})


def get_global_config():
    """获取全局配置字典。
    
    Returns:
        dict: 全局配置字典，可以直接修改来全局改变绘图样式。
    """
    return _GLOBAL_CONFIG


def set_doa_method_style(method_name, color=None, marker=None):
    """设置特定DOA估计方法的样式。
    
    Args:
        method_name (str): DOA估计方法名称。
        color (str, optional): 颜色字符串，如'b', 'g', 'r'等。
        marker (str, optional): 标记字符串，如'x', 'o', 's'等。
    """
    if method_name not in _GLOBAL_CONFIG['doa_method_styles']:
        _GLOBAL_CONFIG['doa_method_styles'][method_name] = {}
    if color is not None:
        _GLOBAL_CONFIG['doa_method_styles'][method_name]['color'] = color
    if marker is not None:
        _GLOBAL_CONFIG['doa_method_styles'][method_name]['marker'] = marker


def get_doa_method_style(method_name, index=0):
    """获取特定DOA估计方法的样式。
    
    Args:
        method_name (str): DOA估计方法名称。
        index (int, optional): 当方法名不在配置中时，使用此索引从默认列表中获取样式。
            默认值为0。
    
    Returns:
        tuple: (color, marker) 元组。
    """
    # 检查是否为已知方法
    config = _GLOBAL_CONFIG['doa_method_styles']
    default_colors = _GLOBAL_CONFIG['default_colors']
    default_markers = _GLOBAL_CONFIG['default_markers']
    
    if method_name in config:
        style = config[method_name]
        color = style.get('color', default_colors[index % len(default_colors)])
        marker = style.get('marker', default_markers[index % len(default_markers)])
        return (color, marker)
    else:
        # 未知方法，使用默认样式
        return (
            default_colors[index % len(default_colors)],
            default_markers[index % len(default_markers)]
        )


def plot_metric_vs_parameter(parameter_values: np.ndarray, results: Dict[str, np.ndarray], 
                            parameter_name: str, metric_name: str, parameter_unit: str = '', 
                            metric_unit: str = '', show_crb: bool = False, 
                            crb_values: Optional[np.ndarray] = None, crb_label: str = 'CRB',
                            ax: Optional[plt.Axes] = None):
    """绘制指标随参数变化的折线图。
    
    Args:
        parameter_values (np.ndarray): 参数值数组。
        results (Dict[str, np.ndarray]): 不同算法的指标结果字典，键为算法名称，值为对应指标值数组。
        parameter_name (str): 参数名称，用于x轴标签。
        metric_name (str): 指标名称，用于y轴标签和图例。
        parameter_unit (str, optional): 参数单位，用于x轴标签。默认值为空字符串。
        metric_unit (str, optional): 指标单位，用于y轴标签。默认值为空字符串。
        show_crb (bool, optional): 是否显示CRB曲线。默认值为False。
        crb_values (Optional[np.ndarray], optional): CRB值数组。默认值为None。
        crb_label (str, optional): CRB曲线的图例标签。默认值为'CRB'。
        ax (Optional[plt.Axes], optional): 外部提供的matplotlib轴对象。如果为None，将创建新图。
            默认值为None。
    """
    # 创建或使用提供的轴
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
        show_plot = True
    else:
        show_plot = False
    
    # 设置x轴标签
    xlabel = f'{parameter_name}'
    if parameter_unit:
        xlabel += f' ({parameter_unit})'
    
    # 设置y轴标签
    ylabel = f'{metric_name}'
    if metric_unit:
        ylabel += f' ({metric_unit})'
    
    # 绘制CRB曲线（如果需要）
    if show_crb and crb_values is not None:
        ax.semilogy(parameter_values, crb_values, '--k', linewidth=2, label=crb_label)
    
    # 绘制每个算法的曲线，使用与DOA方法对应的颜色和标记
    for i, (algorithm, metric_values) in enumerate(results.items()):
        # 从算法名称中提取方法名（去除末尾数字）
        method_name = algorithm
        # 处理类似'RootMUSIC1D'的情况
        if method_name.endswith('1D'):
            method_name = method_name[:-2]
        # 处理类似'CoarrayACMBuilder1D'的情况
        if method_name.endswith('Builder'):
            method_name = method_name[:-7]
        
        # 获取对应的颜色和标记
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
    """绘制真实角度与估计角度的散点图。
    
    Args:
        true_angles (np.ndarray): 真实角度数组，形状为(n_monte_carlo, n_sources)或(n_sources,)。
        estimates (np.ndarray): 估计角度数组，形状为(n_monte_carlo, n_sources)。
        algorithm_name (str): 算法名称，用于标题。
        angle_unit (str, optional): 角度单位，'rad'或'deg'。默认值为'rad'。
        ax (Optional[plt.Axes], optional): 外部提供的matplotlib轴对象。如果为None，将创建新图。
            默认值为None。
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
    
    # 创建或使用提供的轴
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
        show_plot = True
    else:
        show_plot = False
    
    # 绘制每个信源的散点图
    for i in range(n_sources):
        # 对于散点图，我们使用与信源索引对应的默认样式
        # 因为这是单个算法的不同信源
        color, marker = get_doa_method_style(algorithm_name, i)
        ax.scatter(true_angles[:, i], estimates[:, i], color=color, marker=marker, 
                   s=50, alpha=0.6, label=f'Source {i+1}')
    
    # 绘制理想线（y = x）
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
    """绘制估计误差的CDF（累积分布函数）图。
    
    支持两种模式：
    1. 单个算法模式：绘制单个算法的CDF
    2. 多算法比较模式：绘制多个算法的CDF比较
    
    Args:
        estimates (Union[np.ndarray, Dict[str, np.ndarray]]): 
            - 单个算法：估计角度数组，形状为(n_monte_carlo, n_sources)。
            - 多算法比较：字典，键为算法名称，值为对应的估计角度数组。
        true_angles (np.ndarray): 真实角度数组，形状为(n_monte_carlo, n_sources)或(n_sources,)。
        algorithm_name (Union[str, None], optional): 
            - 单个算法：算法名称，用于标题和图例。
            - 多算法比较：None，忽略此参数。
            默认值为None。
        metric_name (str, optional): 误差指标名称，用于x轴标签。默认值为'Error'。
        metric_unit (str, optional): 误差单位，用于x轴标签。默认值为'rad'。
        metric_type (str, optional): 误差类型，可选值：
            - 'absolute': 绝对误差
            - 'rms': 均方根误差
            - 'mae': 平均绝对误差
            默认值为'absolute'。
        ax (Optional[plt.Axes], optional): 外部提供的matplotlib轴对象。如果为None，将创建新图。
            默认值为None。
    """
    # 创建或使用提供的轴
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
        show_plot = True
    else:
        show_plot = False
    
    # 确保true_angles形状正确
    if true_angles.ndim == 1:
        # 单个算法情况
        if isinstance(estimates, np.ndarray):
            true_angles_repeated = np.tile(true_angles, (estimates.shape[0], 1))
        # 多算法情况，获取第一个算法的形状
        else:
            first_alg = next(iter(estimates.keys()))
            true_angles_repeated = np.tile(true_angles, (estimates[first_alg].shape[0], 1))
    else:
        true_angles_repeated = true_angles
    
    # 处理单个算法情况
    if isinstance(estimates, np.ndarray):
        if algorithm_name is None:
            algorithm_name = 'Algorithm'
        
        # 计算误差
        if metric_type == 'absolute':
            errors = np.abs(estimates - true_angles_repeated)
        elif metric_type == 'rms':
            errors = np.sqrt(np.mean(np.square(estimates - true_angles_repeated), axis=1))
            errors = errors[:, np.newaxis]  # 转为(n_monte_carlo, 1)
        elif metric_type == 'mae':
            errors = np.mean(np.abs(estimates - true_angles_repeated), axis=1)
            errors = errors[:, np.newaxis]  # 转为(n_monte_carlo, 1)
        else:
            raise ValueError(f"Unknown metric_type: {metric_type}")
        
        # 转换为度（如果需要）
        if metric_unit == 'deg':
            if metric_type in ['rms', 'mae']:
                errors = np.rad2deg(errors)
            else:
                errors = np.rad2deg(errors)
        
        n_sources = errors.shape[1]
        
        # 绘制每个信源的CDF
        for i in range(n_sources):
            # 获取对应的颜色和标记
            color, _ = get_doa_method_style(algorithm_name, i)
            # 提取第i个信源的误差
            source_errors = errors[:, i]
            # 排序误差
            sorted_errors = np.sort(source_errors)
            # 计算CDF值
            cdf = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
            # 绘制CDF
            ax.plot(sorted_errors, cdf, '-', color=color, linewidth=2, 
                    label=f'{algorithm_name} - Source {i+1}')
    # 处理多算法比较情况
    elif isinstance(estimates, dict):
        # 遍历每个算法
        for i, (alg_name, alg_estimates) in enumerate(estimates.items()):
            # 确保真实角度形状与当前算法估计值一致
            if true_angles.ndim == 1:
                current_true_angles = np.tile(true_angles, (alg_estimates.shape[0], 1))
            else:
                current_true_angles = true_angles
            
            # 计算误差
            if metric_type == 'absolute':
                errors = np.abs(alg_estimates - current_true_angles)
                # 对每个信源单独处理
                n_sources = errors.shape[1]
                for j in range(n_sources):
                    source_errors = errors[:, j]
                    sorted_errors = np.sort(source_errors)
                    cdf = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
                    color, _ = get_doa_method_style(alg_name, j)
                    ax.plot(sorted_errors, cdf, '-', color=color, linewidth=2, 
                            label=f'{alg_name} - Source {j+1}')
            elif metric_type in ['rms', 'mae']:
                # 计算每个蒙特卡洛样本的整体指标
                if metric_type == 'rms':
                    # 每个样本的RMS（所有信源的均方根）
                    sample_errors = np.sqrt(np.mean(np.square(alg_estimates - current_true_angles), axis=1))
                else:  # mae
                    # 每个样本的MAE（所有信源的平均绝对误差）
                    sample_errors = np.mean(np.abs(alg_estimates - current_true_angles), axis=1)
                
                # 转换为度（如果需要）
                if metric_unit == 'deg':
                    sample_errors = np.rad2deg(sample_errors)
                
                # 排序误差
                sorted_errors = np.sort(sample_errors)
                # 计算CDF值
                cdf = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
                # 获取对应的颜色和标记
                color, marker = get_doa_method_style(alg_name, i)
                # 绘制CDF
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

