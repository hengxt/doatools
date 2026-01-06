import inspect
import multiprocessing
import time
import numpy as np
from tqdm import tqdm
from .crb import crb_det_farfield_1d, crb_sto_farfield_1d, crb_stouc_farfield_1d
from ..model.arrays import ArrayDesign
from ..model.signals import ComplexStochasticSignal
from ..model.sources import FarField1DSourcePlacement
from ..model import get_narrowband_snapshots


class PerformanceResult:
    """Encapsulates performance evaluation results.
    
    Attributes:
        snr (float): Signal-to-noise ratio in dB.
        n_snapshots (int): Number of snapshots.
        n_monte_carlo (int): Number of Monte Carlo simulations.
        crb_values (dict): Dictionary containing different types of CRB values, 
            where keys are CRB types and values are CRB values.
        estimator_results (dict): Results for different estimators, where keys are 
            estimator names and values are dictionaries containing different metrics.
        estimator_times (dict): Computation time for each estimator in seconds.
    """

    def __init__(self, snr, n_snapshots, n_monte_carlo):
        self.snr = snr
        self.n_snapshots = n_snapshots
        self.n_monte_carlo = n_monte_carlo
        self.crb_values = {}
        self.estimator_results = {}
        self.estimator_times = {}
        self.sample_estimates = {}

    def add_crb(self, crb_type, value):
        """Adds a CRB value.
        
        Args:
            crb_type (str): Type of CRB.
            value (float): CRB value.
        """
        self.crb_values[crb_type] = value

    def add_estimator_result(self, estimator_name, metric_results, sample_estimates=None, computation_time=0.0):
        """Adds estimator results.
        
        Args:
            estimator_name (str): Name of the estimator.
            metric_results (dict): Metric results, where keys are metric names and 
                values are metric values.
            sample_estimates (array, optional): Array of sample estimates with shape 
                (n_runs, n_sources).
            computation_time (float, optional): Computation time for this estimator in seconds.
        """
        self.estimator_results[estimator_name] = metric_results
        self.estimator_times[estimator_name] = computation_time
        if sample_estimates is not None:
            self.sample_estimates[estimator_name] = sample_estimates

    def __str__(self):
        """Returns a string representation of the results."""
        s = f"Performance Result (SNR: {self.snr} dB, Snapshots: {self.n_snapshots}, Monte Carlo: {self.n_monte_carlo})\n"
        s += "=" * 70 + "\n"

        s += "CRB Values:\n"
        for crb_type, value in self.crb_values.items():
            s += f"  {crb_type.upper():<15}: {value:12.6e} rad²\n"
        s += "\n"

        s += "Estimator Results:\n"
        for estimator_name, metric_results in self.estimator_results.items():
            s += f"  {estimator_name}:\n"
            for metric, value in metric_results.items():
                if metric in ['bias', 'mae', 'rmse']:
                    unit = "rad"
                elif metric == 'mse':
                    unit = "rad²"
                else:
                    unit = ""
                s += f"    {metric.upper():<15}: {value:12.6e} {unit}\n"
            # Add estimator computation time
            s += f"    {'COMPUTATION TIME':<15}: {self.estimator_times[estimator_name]:12.3f} seconds\n"
        s += "\n"
        return s

    def __repr__(self):
        """Returns a repr representation of the results."""
        return self.__str__()


def _single_monte_carlo_run(estimator, array, sources, n_snapshots, power_source, power_noise, run_id):
    """Performs a single Monte Carlo run for a given estimator.
    
    Args:
        estimator: The DOA estimator instance.
        array: The array design.
        sources: The source locations.
        n_snapshots: Number of snapshots.
        power_source: Source power.
        power_noise: Noise power.
        run_id: The run identifier (unused, but required for parallel processing).
        
    Returns:
        tuple: (estimated_locations, run_time) where estimated_locations is numpy.ndarray if resolved, otherwise None,
               and run_time is the time taken for this run in seconds.
    """
    source_signal = ComplexStochasticSignal(sources.size, power_source)
    noise_signal = ComplexStochasticSignal(array.size, power_noise)
    y, Ry = get_narrowband_snapshots(
        array, sources, estimator._wavelength, source_signal, noise_signal,
        n_snapshots, return_covariance=True
    ) 

    run_start_time = time.time()
    # execute DOA estimation
    sig = inspect.signature(estimator.estimate)
    params = list(sig.parameters.keys())
    if len(params) >= 4 or 'd0' in params:
        resolved, estimates = estimator.estimate(Ry, sources.size, array.d0[0])
    else:
        resolved, estimates = estimator.estimate(Ry, sources.size)
    run_time = time.time() - run_start_time

    if resolved:
        return estimates.locations, run_time
    else:
        return None, run_time


class DOAPerformanceEvaluator:
    """DOA performance evaluator for assessing DOA estimation algorithms under different conditions.
    
    This evaluator accepts user-specified parameters, performs Monte Carlo simulations,
    computes evaluation metrics (MSE, RMSE) and theoretical performance bounds (CRLB),
    and returns the results along with computation time.
    """

    def __init__(self, array, sources, snr, n_snapshots, n_monte_carlo,
                 estimators, crb_types=None, metrics=None, save_sample_estimates=False, n_jobs=1):
        """Initializes the performance evaluator.
        
        Args:
            array (~doatools.model.arrays.ArrayDesign): Array design.
            sources (~doatools.model.sources.FarField1DSourcePlacement): Source locations.
            snr (float): Signal-to-noise ratio in dB.
            n_snapshots (int): Number of snapshots.
            n_monte_carlo (int): Number of Monte Carlo simulations.
            estimators (dict or list or object): DOA estimator instance, list of instances, 
                or dictionary of estimators.
                If a dictionary, keys are estimator names and values are estimator instances.
                If a list, estimator class names are used as names.
                If a single instance, its class name is used as the name.
            crb_types (list or str, optional): List of CRLB types or a single type.
                Possible values: 'sto' (stochastic CRB), 'det' (deterministic CRB), 
                'stouc' (stochastic uncorrelated CRB). Default value is ['sto'].
            metrics (list or str, optional): List of evaluation metrics or a single metric.
                Possible values: 'mse' (mean squared error), 'rmse' (root mean squared error).
                Default value is ['mse'].
            save_sample_estimates (bool, optional): Whether to save sample estimates. 
                Default value is False.
            n_jobs (int, optional): Number of parallel jobs to run. Default is 1 (no parallelism).
                If -1, use all available CPUs.
        """
        self.array = array
        self.sources = sources
        self.snr = snr
        self.n_snapshots = n_snapshots
        self.n_monte_carlo = n_monte_carlo
        self.n_jobs = n_jobs

        self.estimators = self._process_estimators(estimators)
        if crb_types is None:
            crb_types = ['sto']
        elif isinstance(crb_types, str):
            crb_types = [crb_types]
        self.crb_types = crb_types
        if metrics is None:
            metrics = ['mse']
        elif isinstance(metrics, str):
            metrics = [metrics]
        self.metrics = metrics
        self._validate_inputs()
        self.save_sample_estimates = save_sample_estimates
        self._precompute_parameters()

    @staticmethod
    def _process_estimators(estimators):
        """Processes the estimators parameter and converts it to a dictionary.
        
        Args:
            estimators (dict or list or object): Estimator instance, list of instances, or dictionary.
        
        Returns:
            dict: Dictionary of estimators, where keys are estimator names and values are instances.
        """
        if isinstance(estimators, dict):
            return estimators
        elif isinstance(estimators, list):
            return {estimator.__class__.__name__: estimator for estimator in estimators}
        else:
            return {estimators.__class__.__name__: estimators}

    def _validate_inputs(self):
        """Validates the input parameters."""
        if not isinstance(self.array, ArrayDesign):
            raise ValueError('array must be an instance of ArrayDesign')

        if not isinstance(self.sources, FarField1DSourcePlacement):
            raise ValueError('sources must be an instance of FarField1DSourcePlacement')

        # validate each estimator
        for name, estimator in self.estimators.items():
            if not hasattr(estimator, 'estimate') or not callable(estimator.estimate):
                raise ValueError(f'estimator {name} must have a callable estimate method')

            if not hasattr(estimator, '_wavelength'):
                raise ValueError(f'estimator {name} must have a _wavelength attribute')

        # validate CRB types
        valid_crb_types = ['sto', 'det', 'stouc']
        for crb_type in self.crb_types:
            if crb_type not in valid_crb_types:
                raise ValueError(f'crb_type {crb_type} must be one of: {valid_crb_types}')

        # validate metrics
        valid_metrics = ['bias', 'mae', 'mse', 'rmse']
        for metric in self.metrics:
            if metric not in valid_metrics:
                raise ValueError(f'metric {metric} must be one of: {valid_metrics}')

        if self.n_monte_carlo <= 0:
            raise ValueError('n_monte_carlo must be positive')

        if self.n_snapshots <= 0:
            raise ValueError('n_snapshots must be positive')

    def _precompute_parameters(self):
        """Precomputes some parameters."""
        self.power_source = 1.0
        self.power_noise = self.power_source / (10 ** (self.snr / 10))
        self.source_signal = ComplexStochasticSignal(self.sources.size, self.power_source)
        self.noise_signal = ComplexStochasticSignal(self.array.size, self.power_noise)

    def _compute_crb(self, crb_type, wavelength):
        """Computes the theoretical performance bound (CRLB).
        
        Args:
            crb_type (str): CRB type.
            wavelength (float): Wavelength.
        
        Returns:
            float: CRB value.
        """
        crb = 0.0
        if crb_type == 'sto':
            crb = crb_sto_farfield_1d(self.array, self.sources, wavelength,
                                      self.power_source, self.power_noise, self.n_snapshots,
                                      return_mode='mean_diag')
        elif crb_type == 'det':
            # for deterministic CRB, we need the source covariance matrix
            # use identity matrix as approximation
            Rs = np.eye(self.sources.size) * self.power_source
            crb = crb_det_farfield_1d(self.array, self.sources, wavelength,
                                      Rs, self.power_noise, self.n_snapshots,
                                      return_mode='mean_diag')
        elif crb_type == 'stouc':
            crb = crb_stouc_farfield_1d(self.array, self.sources, wavelength,
                                        self.power_source, self.power_noise, self.n_snapshots,
                                        return_mode='mean_diag')
        return crb

    def evaluate(self, custom_metrics=None, verbose=0):
        """Performs performance evaluation.
        
        Args:
            custom_metrics (dict, optional): Dictionary of custom evaluation metrics, 
                where keys are metric names and values are functions that accept two parameters 
                (estimated locations and true locations) and return a metric value.
                For example: {'mae': lambda est, true: np.mean(np.abs(est - true))}
            verbose (int, optional): Verbosity level. 0 for silent, 1 for progress bar.
        
        Returns:
            PerformanceResult: Evaluation result object.
        """
        start_time = time.time()

        result = PerformanceResult(
            snr=self.snr,
            n_snapshots=self.n_snapshots,
            n_monte_carlo=self.n_monte_carlo
        )

        # compute all CRB values
        reference_wavelength = next(iter(self.estimators.values()))._wavelength
        for crb_type in self.crb_types:
            crb_value = self._compute_crb(crb_type, reference_wavelength)
            result.add_crb(crb_type, crb_value)

        # handle custom metrics
        if custom_metrics is None:
            custom_metrics = {}

        # evaluate each estimator
        tbar = tqdm(total=len(self.estimators) * self.n_monte_carlo, desc='Evaluating estimators', disable=verbose < 1)

        # Determine the number of processes to use
        if self.n_jobs == -1:
            n_processes = multiprocessing.cpu_count()
        else:
            n_processes = max(1, self.n_jobs)

        for estimator_name, estimator in self.estimators.items():
            metric_results = {}
            all_estimates = []
            all_run_times = []

            if n_processes > 1:
                # Use parallel processing
                with multiprocessing.Pool(processes=n_processes) as pool:
                    # Create a list of arguments for each run
                    args_list = [(estimator, self.array, self.sources, self.n_snapshots,
                                  self.power_source, self.power_noise, i) for i in range(self.n_monte_carlo)]
                    # Execute in parallel
                    results = pool.starmap(_single_monte_carlo_run, args_list)
                    # Collect valid results and run times
                    for res, run_time in results:
                        all_run_times.append(run_time)
                        if res is not None:
                            all_estimates.append(res)
                        tbar.update(1)
            else:
                for _ in range(self.n_monte_carlo):
                    res, run_time = _single_monte_carlo_run(estimator, self.array, self.sources, self.n_snapshots,
                                                          self.power_source, self.power_noise, 0)
                    all_run_times.append(run_time)
                    if res is not None:
                        all_estimates.append(res)
                    tbar.update(1)

            # compute basic statistics
            if len(all_estimates) > 0:
                all_estimates = np.array(all_estimates)
                true_locations = self.sources.locations

                biases = np.mean(all_estimates - true_locations, axis=0)
                avg_bias = np.mean(np.abs(biases))

                mse_per_source = np.mean((all_estimates - true_locations) ** 2, axis=0)
                avg_mse = np.mean(mse_per_source)

                avg_rmse = np.sqrt(avg_mse)

                mae_per_source = np.mean(np.abs(all_estimates - true_locations), axis=0)
                avg_mae = np.mean(mae_per_source)
            else:
                avg_bias = np.pi
                avg_mse = np.pi ** 2
                avg_rmse = np.pi
                avg_mae = np.pi

            # save all metrics
            metric_values = {
                'bias': avg_bias,
                'mae': avg_mae,
                'mse': avg_mse,
                'rmse': avg_rmse
            }

            # save custom metrics
            for metric in self.metrics:
                if metric in metric_values:
                    metric_results[metric] = metric_values[metric]

            if len(all_estimates) > 0:
                for custom_name, custom_func in custom_metrics.items():
                    custom_per_source = []
                    true_locations = self.sources.locations
                    for i in range(true_locations.size):
                        custom_value = custom_func(all_estimates[:, i], true_locations[i])
                        custom_per_source.append(custom_value)
                    metric_results[custom_name] = np.mean(custom_per_source)
            else:
                for custom_name in custom_metrics:
                    metric_results[custom_name] = np.pi

            # Calculate total estimator time as sum of all run times
            estimator_time = np.mean(all_run_times)

            sample_estimates = all_estimates if (self.save_sample_estimates and len(all_estimates) > 0) else None
            result.add_estimator_result(estimator_name, metric_results, sample_estimates, estimator_time)
        tbar.close()
        result.computation_time = time.time() - start_time
        return result


def evaluate_performance(array, sources, snr, n_snapshots, n_monte_carlo,
                         estimators, crb_types=None, metrics=None, custom_metrics=None,
                         save_sample_estimates=False, verbose=0, n_jobs=1):
    """Performance evaluation function for quickly assessing DOA algorithm performance.
    
    This function is a simplified interface for the DOAPerformanceEvaluator class,
    designed for easy direct use by users.
    
    Args:
        array (~doatools.model.arrays.ArrayDesign): Array design.
        sources (~doatools.model.sources.FarField1DSourcePlacement): Source locations.
        snr (float): Signal-to-noise ratio in dB.
        n_snapshots (int): Number of snapshots.
        n_monte_carlo (int): Number of Monte Carlo simulations.
        estimators (dict or list or object): DOA estimator instance, list of instances, or dictionary.
        crb_types (list or str, optional): List of CRLB types or a single type.
            Possible values: 'sto' (stochastic CRB), 'det' (deterministic CRB), 
            'stouc' (stochastic uncorrelated CRB). Default value is ['sto'].\n        metrics (list or str, optional): List of evaluation metrics or a single metric.
            Possible values: 'bias' (bias), 'mae' (mean absolute error), 
            'mse' (mean squared error), 'rmse' (root mean squared error).
            Default value is ['mse'].\n        custom_metrics (dict, optional): Dictionary of custom evaluation metrics.
        metrics
        custom_metrics
        save_sample_estimates (bool, optional): Whether to save sample estimates.
            Default value is False.
        verbose (int, optional): Verbosity level. 0 for silent, 1 for progress bar.
        n_jobs (int, optional): Number of parallel jobs to run. Default is 1 (no parallelism).
            If -1, use all available CPUs.
    
    Returns:
        PerformanceResult: Evaluation result object.
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
        save_sample_estimates=save_sample_estimates,
        n_jobs=n_jobs
    )
    return evaluator.evaluate(custom_metrics, verbose=verbose)
