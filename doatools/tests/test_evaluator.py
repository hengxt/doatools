import numpy as np
import unittest
from unittest.mock import MagicMock, patch
from doatools.performance.evaluator import (
    PerformanceResult,
    _single_monte_carlo_run,
    DOAPerformanceEvaluator,
    evaluate_performance
)
from doatools.model.arrays import UniformLinearArray
from doatools.model.sources import FarField1DSourcePlacement

class TestPerformanceResult(unittest.TestCase):
    
    def test_initialization(self):
        """Test if PerformanceResult initializes correctly."""
        result = PerformanceResult(snr=10, n_snapshots=100, n_monte_carlo=50)
        assert result.snr == 10
        assert result.n_snapshots == 100
        assert result.n_monte_carlo == 50
        assert isinstance(result.crb_values, dict)
        assert isinstance(result.estimator_results, dict)
        assert isinstance(result.estimator_times, dict)
        assert isinstance(result.sample_estimates, dict)
    
    def test_add_crb(self):
        """Test if add_crb correctly adds CRB values."""
        result = PerformanceResult(snr=10, n_snapshots=100, n_monte_carlo=50)
        result.add_crb('sto', 0.1)
        assert 'sto' in result.crb_values
        assert result.crb_values['sto'] == 0.1
    
    def test_add_estimator_result(self):
        """Test if add_estimator_result correctly adds estimator results."""
        result = PerformanceResult(snr=10, n_snapshots=100, n_monte_carlo=50)
        metric_results = {'mse': 0.2, 'rmse': 0.447}
        sample_estimates = np.array([[0.1, 0.2], [0.11, 0.21]])
        
        result.add_estimator_result('MUSIC', metric_results, sample_estimates, 0.5)
        assert 'MUSIC' in result.estimator_results
        assert result.estimator_results['MUSIC'] == metric_results
        assert 'MUSIC' in result.sample_estimates
        assert np.array_equal(result.sample_estimates['MUSIC'], sample_estimates)
        assert result.estimator_times['MUSIC'] == 0.5
    
    def test_str_method(self):
        """Test if __str__ method returns a string representation."""
        result = PerformanceResult(snr=10, n_snapshots=100, n_monte_carlo=50)
        result.add_crb('sto', 0.1)
        result.add_estimator_result('MUSIC', {'mse': 0.2})
        
        str_repr = str(result)
        assert isinstance(str_repr, str)
        assert 'Performance Result' in str_repr
        assert 'CRB Values' in str_repr
        assert 'Estimator Results' in str_repr

class TestDOAPerformanceEvaluator:
    
    def test_process_estimators(self):
        """Test if _process_estimators correctly processes different estimator inputs."""
        # Create mock estimator
        mock_estimator = MagicMock()
        mock_estimator.__class__.__name__ = 'MockEstimator'
        
        # Test with single estimator
        estimators = DOAPerformanceEvaluator._process_estimators(mock_estimator)
        assert isinstance(estimators, dict)
        assert 'MockEstimator' in estimators
        
        # Test with list of estimators
        estimators = DOAPerformanceEvaluator._process_estimators([mock_estimator, mock_estimator])
        assert isinstance(estimators, dict)
        assert len(estimators) == 2
        
        # Test with dictionary
        estimators_dict = {'Estimator1': mock_estimator, 'Estimator2': mock_estimator}
        estimators = DOAPerformanceEvaluator._process_estimators(estimators_dict)
        assert isinstance(estimators, dict)
        assert set(estimators.keys()) == set(estimators_dict.keys())
    
    def test_validate_inputs(self):
        """Test if _validate_inputs correctly validates inputs."""
        # Create valid inputs
        array = UniformLinearArray(5, 0.5)
        sources = FarField1DSourcePlacement(np.array([0.1, 0.2]))
        mock_estimator = MagicMock()
        mock_estimator.estimate = MagicMock(return_value=(True, sources))
        mock_estimator._wavelength = 1.0
        
        # Test with valid inputs
        evaluator = DOAPerformanceEvaluator(
            array=array,
            sources=sources,
            snr=10,
            n_snapshots=100,
            n_monte_carlo=50,
            estimators=mock_estimator
        )
        # If no exception is raised, the test passes
        
        # Test with invalid estimator (no estimate method)
        invalid_estimator = MagicMock()
        invalid_estimator._wavelength = 1.0
        try:
            evaluator = DOAPerformanceEvaluator(
                array=array,
                sources=sources,
                snr=10,
                n_snapshots=100,
                n_monte_carlo=50,
                estimators=invalid_estimator
            )
            assert False, "Should have raised ValueError"
        except ValueError:
            pass
        
        # Test with invalid CRB type
        try:
            evaluator = DOAPerformanceEvaluator(
                array=array,
                sources=sources,
                snr=10,
                n_snapshots=100,
                n_monte_carlo=50,
                estimators=mock_estimator,
                crb_types='invalid_crb'
            )
            assert False, "Should have raised ValueError"
        except ValueError:
            pass
        
        # Test with invalid metric
        try:
            evaluator = DOAPerformanceEvaluator(
                array=array,
                sources=sources,
                snr=10,
                n_snapshots=100,
                n_monte_carlo=50,
                estimators=mock_estimator,
                metrics='invalid_metric'
            )
            assert False, "Should have raised ValueError"
        except ValueError:
            pass
    
    @patch('doatools.performance.evaluator.crb_sto_farfield_1d')
    def test_compute_crb(self, mock_crb_sto):
        """Test if _compute_crb correctly computes CRB values."""
        # Set up mock
        mock_crb_sto.return_value = 0.1
        
        # Create evaluator
        array = UniformLinearArray(5, 0.5)
        sources = FarField1DSourcePlacement(np.array([0.1, 0.2]))
        mock_estimator = MagicMock()
        mock_estimator.estimate = MagicMock(return_value=(True, sources))
        mock_estimator._wavelength = 1.0
        
        evaluator = DOAPerformanceEvaluator(
            array=array,
            sources=sources,
            snr=10,
            n_snapshots=100,
            n_monte_carlo=50,
            estimators=mock_estimator,
            crb_types=['sto']
        )
        
        # Test compute_crb
        crb_value = evaluator._compute_crb('sto', 1.0)
        assert crb_value == 0.1
        mock_crb_sto.assert_called_once()
    
    @patch('doatools.performance.evaluator._single_monte_carlo_run')
    @patch('doatools.performance.evaluator.crb_sto_farfield_1d')
    def test_evaluate(self, mock_crb_sto, mock_monte_carlo_run):
        """Test if evaluate method runs without errors."""
        # Set up mocks
        mock_crb_sto.return_value = 0.1
        mock_monte_carlo_run.return_value = (np.array([0.1, 0.2]), 0.01)
        
        # Create evaluator
        array = UniformLinearArray(5, 0.5)
        sources = FarField1DSourcePlacement(np.array([0.1, 0.2]))
        mock_estimator = MagicMock()
        mock_estimator.estimate = MagicMock(return_value=(True, sources))
        mock_estimator._wavelength = 1.0
        
        evaluator = DOAPerformanceEvaluator(
            array=array,
            sources=sources,
            snr=10,
            n_snapshots=100,
            n_monte_carlo=2,  # Use small number for faster testing
            estimators=mock_estimator,
            crb_types=['sto'],
            metrics=['mse', 'rmse'],
            n_jobs=1  # No parallelism for testing
        )
        
        # Test evaluate method
        result = evaluator.evaluate(verbose=0)
        assert isinstance(result, PerformanceResult)
        assert 'sto' in result.crb_values
        assert 'MockEstimator' in result.estimator_results
    
    @patch('doatools.performance.evaluator.DOAPerformanceEvaluator')
    def test_evaluate_performance(self, mock_evaluator_class):
        """Test if evaluate_performance function works correctly."""
        # Set up mocks
        mock_evaluator = MagicMock()
        mock_result = MagicMock()
        mock_evaluator.evaluate.return_value = mock_result
        mock_evaluator_class.return_value = mock_evaluator
        
        # Create inputs
        array = UniformLinearArray(5, 0.5)
        sources = FarField1DSourcePlacement(np.array([0.1, 0.2]))
        mock_estimator = MagicMock()
        mock_estimator.estimate = MagicMock(return_value=(True, sources))
        mock_estimator._wavelength = 1.0
        
        # Call evaluate_performance
        result = evaluate_performance(
            array=array,
            sources=sources,
            snr=10,
            n_snapshots=100,
            n_monte_carlo=50,
            estimators=mock_estimator
        )
        
        # Check if evaluator was created and evaluate was called
        mock_evaluator_class.assert_called_once()
        mock_evaluator.evaluate.assert_called_once()
        assert result == mock_result
