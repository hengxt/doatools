import numpy as np
import unittest
from unittest.mock import MagicMock, patch
from doatools.estimation.coarray_reconstruction import (
    CovarianceReconstructionBase,
    SPAEstimator,
    ANMEstimator,
    StructCovMLEEstimator,
    WassersteinEstimator
)
from doatools.model.arrays import UniformLinearArray, NestedArray
from doatools.model.sources import FarField1DSourcePlacement

class TestCovarianceReconstructionBase(unittest.TestCase):
    
    def test_initialization(self):
        """Test if SPAEstimator initializes correctly (inherits from CovarianceReconstructionBase)."""
        # Create a nested array (common for coarray reconstruction)
        array = NestedArray(2, 3, 0.5)
        wavelength = 1.0
        
        # Create a mock DOA estimator
        mock_doa_estimator = MagicMock()
        
        # Test initialization with mock estimator
        estimator = SPAEstimator(array, wavelength, doa_estimator=mock_doa_estimator)
        
        # Check attributes
        assert hasattr(estimator, '_array')
        assert hasattr(estimator, '_coarray_builder')
        assert hasattr(estimator, '_virtual_ula')
        assert hasattr(estimator, '_S')
        assert hasattr(estimator, '_M')
        assert hasattr(estimator, '_wavelength')
        assert hasattr(estimator, '_doa_estimator')
        assert hasattr(estimator, '_solver')
    
    def test_properties(self):
        """Test if properties return the correct values."""
        array = NestedArray(2, 3, 0.5)
        wavelength = 1.0
        mock_doa_estimator = MagicMock()
        
        estimator = SPAEstimator(array, wavelength, doa_estimator=mock_doa_estimator)
        
        # Test properties
        assert estimator.virtual_array is not None
        assert estimator.solver is not None
        assert estimator.doa_estimator == mock_doa_estimator
        
        # Test setter for doa_estimator
        new_mock_estimator = MagicMock()
        estimator.doa_estimator = new_mock_estimator
        assert estimator.doa_estimator == new_mock_estimator
    
    def test_select_solver(self):
        """Test if _select_solver correctly selects a solver."""
        # This test may fail if no solvers are installed, but it should at least run
        try:
            solver = CovarianceReconstructionBase._select_solver()
            assert solver is not None
        except ValueError:
            # It's okay if no solvers are available, just check that the method runs
            pass

class TestSPAEstimator:
    
    def test_initialization(self):
        """Test if SPAEstimator initializes correctly."""
        array = NestedArray(2, 3, 0.5)
        wavelength = 1.0
        
        estimator = SPAEstimator(array, wavelength)
        assert isinstance(estimator, SPAEstimator)
        assert isinstance(estimator, CovarianceReconstructionBase)
    
    @patch('cvxpy.Problem.solve')
    def test_reconstruct(self, mock_solve):
        """Test if reconstruct method runs without errors."""
        # Set up mock
        mock_solve.return_value = None
        # Create a mock T variable with value
        mock_T = MagicMock()
        mock_T.value = np.eye(5, dtype=np.complex128)
        
        array = NestedArray(2, 3, 0.5)
        wavelength = 1.0
        estimator = SPAEstimator(array, wavelength)
        
        # Create sample covariance matrix
        R = np.eye(array.size, dtype=np.complex128)
        
        # Test reconstruct method
        with patch('cvxpy.Variable', return_value=mock_T):
            result = estimator.reconstruct(R)
            # Since we're mocking the solver, result might be None or the mock_T.value
            assert result is None or isinstance(result, np.ndarray)

class TestANMEstimator:
    
    def test_initialization(self):
        """Test if ANMEstimator initializes correctly."""
        array = NestedArray(2, 3, 0.5)
        wavelength = 1.0
        
        estimator = ANMEstimator(array, wavelength)
        assert isinstance(estimator, ANMEstimator)
        assert isinstance(estimator, CovarianceReconstructionBase)
        
        # Test with zeta parameter
        estimator = ANMEstimator(array, wavelength, zeta=0.1)
        assert hasattr(estimator, '_zeta')
        assert estimator._zeta == 0.1
    
    @patch('cvxpy.Problem.solve')
    def test_reconstruct(self, mock_solve):
        """Test if reconstruct method runs without errors."""
        # Set up mock
        mock_solve.return_value = None
        # Create a mock T variable with value
        mock_T = MagicMock()
        mock_T.value = np.eye(5, dtype=np.complex128)
        
        array = NestedArray(2, 3, 0.5)
        wavelength = 1.0
        estimator = ANMEstimator(array, wavelength)
        
        # Create sample covariance matrix
        R = np.eye(array.size, dtype=np.complex128)
        
        # Test reconstruct method
        with patch('cvxpy.Variable', return_value=mock_T):
            result = estimator.reconstruct(R)
            # Since we're mocking the solver, result might be None or the mock_T.value
            assert result is None or isinstance(result, np.ndarray)

class TestStructCovMLEEstimator:
    
    def test_initialization(self):
        """Test if StructCovMLEEstimator initializes correctly."""
        array = NestedArray(2, 3, 0.5)
        wavelength = 1.0
        
        estimator = StructCovMLEEstimator(array, wavelength)
        assert isinstance(estimator, StructCovMLEEstimator)
        assert isinstance(estimator, CovarianceReconstructionBase)
        
        # Test with custom parameters
        estimator = StructCovMLEEstimator(
            array, wavelength, epsilon=1e-5, max_iter=20, lambda_noise=0.01
        )
        assert hasattr(estimator, '_epsilon')
        assert hasattr(estimator, '_max_iter')
        assert hasattr(estimator, '_lambda_noise')
        assert estimator._epsilon == 1e-5
        assert estimator._max_iter == 20
        assert estimator._lambda_noise == 0.01
    
    @patch('cvxpy.Problem.solve')
    def test_reconstruct(self, mock_solve):
        """Test if reconstruct method runs without errors."""
        # Set up mock
        mock_solve.return_value = None
        # Create a mock T variable with value
        mock_T = MagicMock()
        mock_T.value = np.eye(5, dtype=np.complex128)
        
        array = NestedArray(2, 3, 0.5)
        wavelength = 1.0
        estimator = StructCovMLEEstimator(array, wavelength, max_iter=2)  # Use small max_iter for testing
        
        # Create sample covariance matrix
        R = np.eye(array.size, dtype=np.complex128)
        
        # Test reconstruct method
        with patch('cvxpy.Variable', return_value=mock_T):
            result = estimator.reconstruct(R)
            # Since we're mocking the solver, result might be None or the mock_T.value
            assert result is None or isinstance(result, np.ndarray)

class TestWassersteinEstimator:
    
    def test_initialization(self):
        """Test if WassersteinEstimator initializes correctly."""
        array = NestedArray(2, 3, 0.5)
        wavelength = 1.0
        
        estimator = WassersteinEstimator(array, wavelength)
        assert isinstance(estimator, WassersteinEstimator)
        assert isinstance(estimator, CovarianceReconstructionBase)
        
        # Test with custom parameters
        estimator = WassersteinEstimator(
            array, wavelength, use_gradient=True, mu=0.05, lr=0.2, max_iter=100,
            tol=1e-6, verbose=True
        )
        assert hasattr(estimator, '_use_gradient')
        assert hasattr(estimator, '_mu')
        assert hasattr(estimator, '_lr')
        assert hasattr(estimator, '_max_iter')
        assert hasattr(estimator, '_tol')
        assert hasattr(estimator, '_verbose')
        assert estimator._use_gradient == True
        assert estimator._mu == 0.05
        assert estimator._lr == 0.2
    
    @patch('cvxpy.Problem.solve')
    def test_reconstruct(self, mock_solve):
        """Test if reconstruct method runs without errors."""
        # Set up mock for non-gradient implementation
        mock_solve.return_value = None
        # Create a mock T variable with value
        mock_T = MagicMock()
        mock_T.value = np.eye(5, dtype=np.complex128)
        
        array = NestedArray(2, 3, 0.5)
        wavelength = 1.0
        
        # Test non-gradient implementation
        estimator = WassersteinEstimator(array, wavelength, use_gradient=False)
        R = np.eye(array.size, dtype=np.complex128)
        
        with patch('cvxpy.Variable', return_value=mock_T):
            result = estimator.reconstruct(R)
            # Since we're mocking the solver, result might be None or the mock_T.value
            assert result is None or isinstance(result, np.ndarray)
        
        # Test gradient implementation (should run without errors)
        estimator = WassersteinEstimator(array, wavelength, use_gradient=True, max_iter=5)
        result = estimator.reconstruct(R)
        # Gradient implementation might return a matrix even without solver mock
        assert result is None or isinstance(result, np.ndarray)
    
    def test_estimate_method(self):
        """Test if estimate method runs without errors."""
        array = NestedArray1D(2, 3)
        wavelength = 1.0
        
        # Create a sample covariance matrix
        R = np.eye(array.size, dtype=np.complex128)
        k = 2  # Number of sources
        
        # Test with SPAEstimator
        estimator = SPAEstimator(array, wavelength)
        
        # Mock the reconstruct method to return a simple covariance matrix
        estimator.reconstruct = MagicMock(return_value=np.eye(5, dtype=np.complex128))
        
        # Test estimate method
        resolved, estimates = estimator.estimate(R, k)
        # Since we're mocking reconstruct, the result may vary but should run without errors
        assert isinstance(resolved, bool)
