import numpy as np
import time
import inspect
from ..model.arrays import ArrayDesign
from ..model.sources import FarField1DSourcePlacement
from ..model.signals import ComplexStochasticSignal
from .crb import crb_det_farfield_1d, crb_sto_farfield_1d, crb_stouc_farfield_1d
from .mse import ecov_music_1d


class PerformanceResult:
    """封装性能评估结果的类。
    
    Attributes:
        snr (float): 信噪比（dB）。
        n_snapshots (int): 快照数。
        n_monte_carlo (int): 蒙特卡洛模拟次数。
        crb_values (dict): 不同类型CRB的值，键为CRB类型，值为CRB值。
        crb_results (dict): 与crb_values相同，为向后兼容而提供的别名。
        estimator_results (dict): 不同估计器的结果，键为估计器名称，
            值为包含不同指标的字典，键为指标名称，值为指标值。
        computation_time (float): 总计算时间（秒）。
    """
    
    def __init__(self, snr, n_snapshots, n_monte_carlo):
        self.snr = snr
        self.n_snapshots = n_snapshots
        self.n_monte_carlo = n_monte_carlo
        self.crb_values = {}
        self.estimator_results = {}
        self.sample_estimates = {}
        self.computation_time = 0.0
    
    def add_crb(self, crb_type, value):
        """添加CRB值。
        
        Args:
            crb_type (str): CRB类型。
            value (float): CRB值。
        """
        self.crb_values[crb_type] = value
    
    def add_estimator_result(self, estimator_name, metric_results, sample_estimates=None):
        """添加估计器结果。
        
        Args:
            estimator_name (str): 估计器名称。
            metric_results (dict): 指标结果，键为指标名称，值为指标值。
            sample_estimates (array, optional): 样本估计值数组，形状为(n_runs, n_sources)。
        """
        self.estimator_results[estimator_name] = metric_results
        if sample_estimates is not None:
            self.sample_estimates[estimator_name] = sample_estimates
    
    def __str__(self):
        """返回结果的字符串表示。"""
        s = f"Performance Result (SNR: {self.snr} dB, Snapshots: {self.n_snapshots}, Monte Carlo: {self.n_monte_carlo})\n"
        s += "=" * 70 + "\n"
        
        # 打印CRB值
        s += "CRB Values:\n"
        for crb_type, value in self.crb_values.items():
            s += f"  {crb_type.upper():<15}: {value:12.6e} rad²\n"
        s += "\n"
        
        # 打印估计器结果
        s += "Estimator Results:\n"
        for estimator_name, metric_results in self.estimator_results.items():
            s += f"  {estimator_name}:\n"
            for metric, value in metric_results.items():
                # 确定单位
                if metric in ['bias', 'mae', 'rmse']:
                    unit = "rad"
                elif metric == 'mse':
                    unit = "rad²"
                else:  # 自定义指标
                    unit = ""
                s += f"    {metric.upper():<15}: {value:12.6e} {unit}\n"
        s += "\n"
        
        # 打印计算时间
        s += f"Total Computation Time: {self.computation_time:.3f} seconds\n"
        
        return s
    
    def __repr__(self):
        """返回结果的repr表示。"""
        return self.__str__()


class DOAPerformanceEvaluator:
    """DOA性能评估器，用于评估不同条件下DOA估计算法的性能。
    
    该评估器接受用户指定的参数，执行蒙特卡洛模拟，计算评估指标（MSE、RMSE）
    和理论性能下界（CRLB），并返回结果和计算时间。
    """
    
    def __init__(self, array, sources, snr, n_snapshots, n_monte_carlo,
                 estimators, crb_types=None, metrics=None, save_sample_estimates=False):
        """初始化性能评估器。
        
        Args:
            array (~doatools.model.arrays.ArrayDesign): 阵列设计。
            sources (~doatools.model.sources.FarField1DSourcePlacement): 信源位置。
            snr (float): 信噪比（dB）。
            n_snapshots (int): 快照数。
            n_monte_carlo (int): 蒙特卡洛模拟次数。
            estimators (dict or list or object): DOA估计算法实例或实例列表或字典。
                如果是字典，键为估计器名称，值为估计器实例；
                如果是列表，使用估计器类名作为名称；
                如果是单个实例，使用其类名作为名称。
            crb_types (list or str, optional): CRLB类型列表或单个类型，
                可选值：'sto'（随机CRB）、'det'（确定性CRB）、'stouc'（随机无相关CRB）。
                默认值为['sto']。
            metrics (list or str, optional): 评估指标列表或单个指标，
                可选值：'mse'（均方误差）、'rmse'（均方根误差）。
                默认值为['mse']。
            save_sample_estimates (bool, optional): 是否保存采样估计值。默认值为False。
        """
        self.array = array
        self.sources = sources
        self.snr = snr
        self.n_snapshots = n_snapshots
        self.n_monte_carlo = n_monte_carlo
        
        # 处理estimators参数
        self.estimators = self._process_estimators(estimators)
        
        # 处理crb_types参数
        if crb_types is None:
            crb_types = ['sto']
        elif isinstance(crb_types, str):
            crb_types = [crb_types]
        self.crb_types = crb_types
        
        # 处理metrics参数
        if metrics is None:
            metrics = ['mse']
        elif isinstance(metrics, str):
            metrics = [metrics]
        self.metrics = metrics
        
        # 验证输入参数
        self._validate_inputs()
        
        # 保存是否保存样本估计值的标志
        self.save_sample_estimates = save_sample_estimates
        
        # 预计算一些参数
        self._precompute_parameters()
    
    def _process_estimators(self, estimators):
        """处理estimators参数，转换为字典格式。
        
        Args:
            estimators (dict or list or object): 估计器实例或实例列表或字典。
        
        Returns:
            dict: 估计器字典，键为估计器名称，值为估计器实例。
        """
        if isinstance(estimators, dict):
            return estimators
        elif isinstance(estimators, list):
            return {estimator.__class__.__name__: estimator for estimator in estimators}
        else:
            return {estimators.__class__.__name__: estimators}
    
    def _validate_inputs(self):
        """验证输入参数的有效性。"""
        if not isinstance(self.array, ArrayDesign):
            raise ValueError('array must be an instance of ArrayDesign')
        
        if not isinstance(self.sources, FarField1DSourcePlacement):
            raise ValueError('sources must be an instance of FarField1DSourcePlacement')
        
        # 验证每个估计器
        for name, estimator in self.estimators.items():
            # 检查estimator是否具有estimate方法
            if not hasattr(estimator, 'estimate') or not callable(estimator.estimate):
                raise ValueError(f'estimator {name} must have a callable estimate method')
            
            # 检查estimator是否具有wavelength属性
            if not hasattr(estimator, '_wavelength'):
                raise ValueError(f'estimator {name} must have a _wavelength attribute')
        
        # 验证CRB类型
        valid_crb_types = ['sto', 'det', 'stouc']
        for crb_type in self.crb_types:
            if crb_type not in valid_crb_types:
                raise ValueError(f'crb_type {crb_type} must be one of: {valid_crb_types}')
        
        # 验证指标
        valid_metrics = ['bias', 'mae', 'mse', 'rmse']
        for metric in self.metrics:
            if metric not in valid_metrics:
                raise ValueError(f'metric {metric} must be one of: {valid_metrics}')
        
        if self.n_monte_carlo <= 0:
            raise ValueError('n_monte_carlo must be positive')
        
        if self.n_snapshots <= 0:
            raise ValueError('n_snapshots must be positive')
    
    def _precompute_parameters(self):
        """预计算一些参数。"""
        # 计算噪声功率
        self.power_source = 1.0  # 归一化源功率
        self.power_noise = self.power_source / (10**(self.snr / 10))
        
        # 初始化信号源和噪声
        self.source_signal = ComplexStochasticSignal(self.sources.size, self.power_source)
        self.noise_signal = ComplexStochasticSignal(self.array.size, self.power_noise)
    
    def _compute_crb(self, crb_type, wavelength):
        """计算理论性能下界（CRLB）。
        
        Args:
            crb_type (str): CRB类型。
            wavelength (float): 波长。
        
        Returns:
            float: CRB值。
        """
        if crb_type == 'sto':
            crb = crb_sto_farfield_1d(self.array, self.sources, wavelength,
                                      self.power_source, self.power_noise, self.n_snapshots,
                                      return_mode='mean_diag')
        elif crb_type == 'det':
            # 对于确定性CRB，我们需要源协方差矩阵
            # 使用单位矩阵作为近似
            Rs = np.eye(self.sources.size) * self.power_source
            crb = crb_det_farfield_1d(self.array, self.sources, wavelength,
                                      Rs, self.power_noise, self.n_snapshots,
                                      return_mode='mean_diag')
        elif crb_type == 'stouc':
            crb = crb_stouc_farfield_1d(self.array, self.sources, wavelength,
                                       self.power_source, self.power_noise, self.n_snapshots,
                                       return_mode='mean_diag')
        return crb
    
    def evaluate(self, custom_metrics=None):
        """执行性能评估。
        
        Args:
            custom_metrics (dict, optional): 自定义评价指标字典，键为指标名称，
                值为接受两个参数（估计位置和真实位置）的函数，返回指标值。
                例如：{'mae': lambda est, true: np.mean(np.abs(est - true))}
        
        Returns:
            PerformanceResult: 评估结果对象。
        """
        start_time = time.time()
        
        # 创建结果对象
        result = PerformanceResult(
            snr=self.snr,
            n_snapshots=self.n_snapshots,
            n_monte_carlo=self.n_monte_carlo
        )
        
        # 计算所有CRB值
        # 注意：CRB计算只需要进行一次，与估计器无关
        # 使用第一个估计器的波长作为参考
        reference_wavelength = next(iter(self.estimators.values()))._wavelength
        for crb_type in self.crb_types:
            crb_value = self._compute_crb(crb_type, reference_wavelength)
            result.add_crb(crb_type, crb_value)
        
        # 处理自定义指标
        if custom_metrics is None:
            custom_metrics = {}
        
        # 对每个估计器执行蒙特卡洛模拟
        for estimator_name, estimator in self.estimators.items():
            # 初始化指标结果字典
            metric_results = {}
            
            # 存储所有成功估计的位置，用于计算偏差和其他指标
            all_estimates = []
            
            # 执行蒙特卡洛模拟
            for _ in range(self.n_monte_carlo):
                # 生成信号和噪声
                S = self.source_signal.emit(self.n_snapshots)
                N = self.noise_signal.emit(self.n_snapshots)
                
                # 生成阵列输出
                A = self.array.steering_matrix(self.sources, estimator._wavelength)
                Y = A @ S + N
                
                # 计算协方差矩阵
                Ry = (Y @ Y.conj().T) / self.n_snapshots
                
                # 执行DOA估计
                # 检查estimate方法的签名，决定是否传递d0参数
                sig = inspect.signature(estimator.estimate)
                params = list(sig.parameters.keys())
                if len(params) >= 4 or 'd0' in params:
                    # 方法接受d0参数，如RootMUSIC1D
                    resolved, estimates = estimator.estimate(Ry, self.sources.size, self.array.d0[0])
                else:
                    # 方法不接受d0参数，如MUSIC
                    resolved, estimates = estimator.estimate(Ry, self.sources.size)
                
                # 保存成功的估计
                if resolved:
                    all_estimates.append(estimates.locations)
            
            # 计算基本统计量
            if len(all_estimates) > 0:
                # 将列表转换为数组 (n_runs, n_sources)
                all_estimates = np.array(all_estimates)
                true_locations = self.sources.locations
                
                # 计算偏差 (每个信源的平均偏差，然后取平均值)
                biases = np.mean(all_estimates - true_locations, axis=0)
                avg_bias = np.mean(np.abs(biases))
                
                # 计算MSE (每个信源的MSE，然后取平均值)
                mse_per_source = np.mean((all_estimates - true_locations)**2, axis=0)
                avg_mse = np.mean(mse_per_source)
                
                # 计算RMSE
                avg_rmse = np.sqrt(avg_mse)
                
                # 计算MAE (每个信源的MAE，然后取平均值)
                mae_per_source = np.mean(np.abs(all_estimates - true_locations), axis=0)
                avg_mae = np.mean(mae_per_source)
            else:
                # 如果没有成功的估计，使用较大的默认值
                avg_bias = np.pi
                avg_mse = np.pi**2
                avg_rmse = np.pi
                avg_mae = np.pi
            
            # 保存内置指标结果
            metric_values = {
                'bias': avg_bias,
                'mae': avg_mae,
                'mse': avg_mse,
                'rmse': avg_rmse
            }
            
            # 计算并保存用户指定的指标
            for metric in self.metrics:
                if metric in metric_values:
                    metric_results[metric] = metric_values[metric]
            
            # 计算并保存自定义指标
            if len(all_estimates) > 0:
                for custom_name, custom_func in custom_metrics.items():
                    # 对每个信源计算自定义指标，然后取平均值
                    custom_per_source = []
                    for i in range(true_locations.size):
                        custom_value = custom_func(all_estimates[:, i], true_locations[i])
                        custom_per_source.append(custom_value)
                    metric_results[custom_name] = np.mean(custom_per_source)
            else:
                # 如果没有成功的估计，使用较大的默认值
                for custom_name in custom_metrics:
                    metric_results[custom_name] = np.pi
            
            # 添加估计器结果，根据配置决定是否包括样本估计值
            sample_estimates = all_estimates if (self.save_sample_estimates and len(all_estimates) > 0) else None
            result.add_estimator_result(estimator_name, metric_results, sample_estimates)
        
        # 计算总时间
        result.computation_time = time.time() - start_time
        
        # 返回结果
        return result


def evaluate_performance(array, sources, snr, n_snapshots, n_monte_carlo,
                         estimators, crb_types=None, metrics=None, custom_metrics=None,
                         save_sample_estimates=False):
    """性能评估函数，用于快速评估DOA算法性能。
    
    该函数是DOAPerformanceEvaluator类的简化接口，方便用户直接调用。
    
    Args:
        array (~doatools.model.arrays.ArrayDesign): 阵列设计。
        sources (~doatools.model.sources.FarField1DSourcePlacement): 信源位置。
        snr (float): 信噪比（dB）。
        n_snapshots (int): 快照数。
        n_monte_carlo (int): 蒙特卡洛模拟次数。
        estimators (dict or list or object): DOA估计算法实例或实例列表或字典。
        crb_types (list or str, optional): CRLB类型列表或单个类型，
            可选值：'sto'（随机CRB）、'det'（确定性CRB）、'stouc'（随机无相关CRB）。
            默认值为['sto']。
        metrics (list or str, optional): 评估指标列表或单个指标，
            可选值：'bias'（偏差）、'mae'（平均绝对误差）、'mse'（均方误差）、'rmse'（均方根误差）。
            默认值为['mse']。
        custom_metrics (dict, optional): 自定义评价指标字典，键为指标名称，
            值为接受两个参数（估计位置和真实位置）的函数，返回指标值。
            例如：{'custom_metric': lambda est, true: np.mean(np.abs(est - true))}
        save_sample_estimates (bool, optional): 是否保存采样估计值。默认值为False。
    
    Returns:
        PerformanceResult: 评估结果对象。
    """
    evaluator = DOAPerformanceEvaluator(
        array=array,
        sources=sources,
        snr=snr,
        n_snapshots=n_snapshots,
        n_monte_carlo=n_monte_carlo,
        estimators=estimators,
        crb_types=crb_types,
        metrics=metrics,
        save_sample_estimates=save_sample_estimates
    )
    return evaluator.evaluate(custom_metrics)
