from abc import ABC, abstractmethod
import numpy as np
from scipy.linalg import inv, sqrtm, toeplitz

import cvxpy as cp

from .core import ensure_covariance_size
from .coarray import CoarrayACMBuilder1D
from .music import MUSIC, RootMUSIC1D


class CovarianceReconstructionBase(ABC):
    """Base class for covariance matrix reconstruction algorithms.
    
    Covariance matrix reconstruction algorithms aim to transform the sample 
    covariance matrix of a sparse array into an augmented covariance matrix 
    that corresponds to a larger virtual uniform linear array, typically 
    based on the difference coarray concept.
    """

    def __init__(self, array, wavelength, doa_estimator=None, search_grid=None, **kwargs):
        self._array = array
        self._coarray_builder = CoarrayACMBuilder1D(array)
        self._virtual_ula = self._coarray_builder.get_virtual_ula()
        self._S = self._coarray_builder.select_matrix
        self._M = self._coarray_builder.mask_matrix
        self._wavelength = wavelength
        self._doa_estimator = doa_estimator
        self._solver = self._select_solver()

    @staticmethod
    def _select_solver():
        """Select a valid solver starting from MOSEK.
        
        Returns:
            str: Selected solver name.
        """
        available_solvers = cp.installed_solvers()
        # Priority list: MOSEK > ECOS > SCS
        for solver in ['MOSEK', 'ECOS', 'SCS']:
            if solver in available_solvers:
                return getattr(cp, solver)
        raise ValueError("No valid solver found.")

    @property
    def virtual_array(self):
        """Retrieves the corresponding virtual uniform linear array.
        
        Returns:
            ~doatools.model.arrays.UniformLinearArray: Virtual uniform linear array.
        """
        return self._virtual_ula

    @property
    def solver(self):
        """Retrieves the selected solver.
        
        Returns:
            str: Selected solver name.
        """
        return self._solver

    @property
    def doa_estimator(self):
        """Retrieves the DOA estimator.
        
        Returns:
            DOA estimator instance or None.
        """
        return self._doa_estimator

    @doa_estimator.setter
    def doa_estimator(self, doa_estimator):
        """Sets the DOA estimator.

        Args:
            doa_estimator (DOA estimator instance or None): DOA estimator.
        """
        self._doa_estimator = doa_estimator

    # @abstractmethod
    def reconstruct(self, R, Y=None, **kwargs):
        """Reconstructs the augmented covariance matrix.
        
        Args:
            R (~numpy.ndarray): Sample covariance matrix of the sparse array.
            Y (~numpy.ndarray): Sample signal of the sparse array.
            
        Returns:
            ~numpy.ndarray: Augmented covariance matrix.
        """
        return R

    def _doa_estimate(self, Ra, k, **kwargs):
        search_grid = kwargs.get('search_grid', None)
        unit = kwargs.get('unit', 'rad')
        # Use the virtual ULA for DOA estimation,  Default to RootMUSIC1D
        if self._doa_estimator is None:
            doa_estimator = RootMUSIC1D(self._wavelength)
            resolved, estimates = doa_estimator.estimate(Ra, k, d0=self._virtual_ula.d0, unit=unit)
            if 'return_spectrum' in kwargs and kwargs['return_spectrum']:
                if search_grid is None:
                    return resolved, estimates, None
                music = MUSIC(self._virtual_ula, self._wavelength, search_grid)
                _, _, spectrum = music.estimate(Ra, k, return_spectrum=True)
                return resolved, estimates, spectrum
            else:
                return resolved, estimates
        elif isinstance(self._doa_estimator, RootMUSIC1D):
            resolved, estimates = self._doa_estimator.estimate(Ra, k, d0=self._virtual_ula.d0, unit=unit)
            return resolved, estimates
        else:
            if hasattr(self._doa_estimator, '_search_grid') and search_grid is None:
                if 'return_spectrum' in kwargs and kwargs['return_spectrum']:
                    return False, None, None
                else:
                    return False, None
            return self._doa_estimator.estimate(Ra, k, **kwargs)


    def estimate(self, R, k, Y=None, **kwargs):
        r"""Estimates the source locations from the given covariance matrix.

        Args:
            R (~numpy.ndarray): Covariance matrix input. The size of R must
                match that of the array design used when creating this
                estimator.
            k (int): Expected number of sources.
            **kwargs: Other keyword arguments.

        Returns:
            A tuple with the following elements.

            * resolved (:class:`bool`): ``True`` if the desired number of
              sources are found. This flag does **not** guarantee that the
              estimated source locations are correct. The estimated source
              locations may be completely wrong!
              If resolved is False, both ``estimates`` and ``spectrum`` will be
              ``None``.
            * estimates (:class:`~doatools.model.sources.SourcePlacement`):
              A :class:`~doatools.model.sources.SourcePlacement` instance of the
              same type as the one used in the search grid (if provided),
              representing the estimated source locations. Will be ``None`` if resolved is
              ``False``.
            * spectrum (:class:`~numpy.ndarray`): An numpy array of the same
              shape of the specified search grid, consisting of values evaluated
              at the grid points. Only present if ``return_spectrum`` is
              ``True`` and a search grid is provided.
        """
        ensure_covariance_size(R, self._array)
        Ra = self.reconstruct(R, Y=Y, **kwargs)
        if Ra is None:
            if 'return_spectrum' in kwargs and kwargs['return_spectrum']:
                return False, None, None
            else:
                return False, None
        return self._doa_estimate(Ra, k, **kwargs)


class DAEstimator(CovarianceReconstructionBase):
    """Creates a source location estimator based on Spatial-smoothing based 
    Augmentation (SPA) algorithm.
    
    The SPA algorithm reconstructs a large covariance matrix corresponding to 
    the difference coarray uniform linear array using semi-definite programming 
    (SDP) approach, and then applies subspace-based methods (such as MUSIC) 
    for DOA estimation.
    
    Args:
        array (~doatools.model.arrays.ArrayDesign): Array design.
        wavelength (float): Wavelength of the carrier wave.
        search_grid (~doatools.estimation.grid.SearchGrid, optional): The search grid 
            used to locate the sources. Defaults to None. Not needed for 
            gridless DOA estimators like RootMUSIC1D and ESPRIT.
        method (str, optional): 'DA' or 'SS'. Defaults to 'da'.
        doa_estimator: DOA estimator instance. Defaults to None (RootMUSIC1D).
        **kwargs: Other keyword arguments.
            
    References:
        [1] Y.I. Abramovich, D.A. Gray, A.Y. Gorokhov, and N.K. Spencer, “Positive-definite toeplitz completion in DOA estimation for nonuniform linear antenna arrays. i. fully augmentable arrays,” IEEE Trans. Signal Process., vol. 46, no. 9, pp. 24582471, 1998.
    """

    def __init__(self, array, wavelength, method: str='da', doa_estimator=None, search_grid=None, **kwargs):
        super().__init__(array, wavelength, doa_estimator, search_grid, **kwargs)
        self._method = 'da'

    def reconstruct(self, R, Y=None, **kwargs):
        """Reconstructs the augmented covariance matrix using Direct Augenment algorithm.

        Args:
            R (~numpy.ndarray): Sample covariance matrix of the sparse array.

        Returns:
            ~numpy.ndarray: Augmented covariance matrix.
        """
        scm_vector = self._coarray_builder.transform(R, self._method)
        return scm_vector


class SPAEstimator(CovarianceReconstructionBase):
    """Creates a source location estimator based on Spatial-smoothing based 
    Augmentation (SPA) algorithm.
    
    The SPA algorithm reconstructs a large covariance matrix corresponding to 
    the difference coarray uniform linear array using semi-definite programming 
    (SDP) approach, and then applies subspace-based methods (such as MUSIC) 
    for DOA estimation.
    
    Args:
        array (~doatools.model.arrays.ArrayDesign): Array design.
        wavelength (float): Wavelength of the carrier wave.
        search_grid (~doatools.estimation.grid.SearchGrid, optional): The search grid 
            used to locate the sources. Defaults to None. Not needed for 
            gridless DOA estimators like RootMUSIC1D and ESPRIT.
        lambda_noise (float, optional): The noise variance, used for the 
                                        SPA_noisevar variant. Defaults to 0
                                        (standard SPA).
        doa_estimator: DOA estimator instance. Defaults to None (RootMUSIC1D).
        **kwargs: Other keyword arguments.
            
    References:
        [1] Z. Yang, L. Xie, and C. Zhang, "A Discretization-Free Sparse and Parametric Approach for Linear Array Signal Processing," *IEEE Transactions on Signal Processing*, vol. 62, no. 19, pp. 4959-4973, Oct. 2014.
    """

    def __init__(self, array, wavelength, lambda_noise: float = 0, doa_estimator=None, search_grid=None, **kwargs):
        super().__init__(array, wavelength, doa_estimator, search_grid, **kwargs)
        self._lambda_noise = lambda_noise

    def reconstruct(self, R, Y=None, **kwargs):
        """Reconstructs the augmented covariance matrix using SPA algorithm.
        
        Args:
            R (~numpy.ndarray): Sample covariance matrix of the sparse array.
            
        Returns:
            ~numpy.ndarray: Augmented covariance matrix.
        """
        ensure_covariance_size(R, self._array)
        S_const = cp.Constant(self._S)

        n, m = self._S.shape
        R_hat = (R + R.T.conj()) / 2
        R_hat_inv = inv(R_hat)
        R_sqrt = cp.Constant(sqrtm(R_hat))
        R_hat_inv = cp.Constant((R_hat_inv + R_hat_inv.T.conj()) / 2)
        R_sqrt = (R_sqrt + R_sqrt.T.conj()) / 2
        T = cp.Variable((m, m), hermitian=True)
        X = cp.Variable((n, n), hermitian=True)
        objective = cp.Minimize(cp.real(cp.trace(X) + cp.trace(R_hat_inv @ S_const @ T @ S_const.T.conj())))
        Z = cp.Constant(np.zeros((n, m), dtype=complex))
        I_n = cp.Constant(np.eye(n))


        if self._lambda_noise > 0:
            lmi_matrix = cp.bmat([
                [X, R_sqrt, Z],
                [R_sqrt.T.conj(), S_const @ T @ S_const.T.conj() + self._lambda_noise * I_n, Z],
                [Z.T.conj(), Z.T.conj(), T]
            ])
        else:
            lmi_matrix = cp.bmat([
                [X, R_sqrt, Z],
                [R_sqrt.T.conj(), S_const @ T @ S_const.T.conj(), Z],
                [Z.T.conj(), Z.T.conj(), T]
            ])
        constraints = [lmi_matrix >> 0]
        for i in range(m - 1):
            for j in range(m - 1):
                constraints.append(T[i, j] == T[i + 1, j + 1])

        prob = cp.Problem(objective, constraints)
        prob.solve(solver=self._solver, verbose=False)
        if prob.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            print(f"CVXPY Solver Error: {prob.status}")
            return None
        first_row = T.value[:, 0]
        T = toeplitz(first_row)
        T = (T + T.T.conj()) / 2
        return T


class ANMEstimator(CovarianceReconstructionBase):
    """Creates a source location estimator based on Augmentation via 
    Nuclear-norm Minimization (ANM) algorithm.
    
    The ANM algorithm reconstructs the augmented covariance matrix by solving 
    a nuclear-norm minimization problem, which exploits the low-rank property 
    of the covariance matrix.
    
    Args:
        array (~doatools.model.arrays.ArrayDesign): Array design.
        wavelength (float): Wavelength of the carrier wave.
        search_grid (~doatools.estimation.grid.SearchGrid, optional): The search grid 
            used to locate the sources. Defaults to None. Not needed for 
            gridless DOA estimators like RootMUSIC1D and ESPRIT.
        zeta (float, optional): Regularization parameter. Defaults to 0.0.
        max_workers (int, optional): Maximum number of workers for parallel 
                                      processing. Defaults to 1.
        doa_estimator: DOA estimator instance. Defaults to None (RootMUSIC1D).
        **kwargs: Other keyword arguments.
            
    References:
        [1] C. Zhou, Y. Gu, X. Fan, Z. Shi, G. Mao, and Y. D. Zhang, "Direction-of-Arrival Estimation for Coprime Array via Virtual Array Interpolation," *IEEE Transactions on Signal Processing*, vol. 66, no. 22, pp. 5956-5971, Nov. 2018.
        [2] X. Wu, W.-P. Zhu, and J. Yan, "A Toeplitz Covariance Matrix Reconstruction Approach for Direction-of-Arrival Estimation," *IEEE Transactions on Vehicular Technology*, vol. 66, no. 9, pp. 8223-8237, Sept. 2017.
    """

    def __init__(self, array, wavelength, zeta: float = 0.0, doa_estimator=None, search_grid=None, **kwargs):
        super().__init__(array, wavelength, doa_estimator, search_grid, **kwargs)
        self._zeta = zeta

    def reconstruct(self, R, Y=None, **kwargs):
        """Reconstructs the augmented covariance matrix using ANM algorithm.
        
        Args:
            R (~numpy.ndarray): Sample covariance matrix of the sparse array.
            
        Returns:
            ~numpy.ndarray: Augmented covariance matrix.
        """
        ensure_covariance_size(R, self._array)
        M_const = cp.Constant(self._M)
        S_const = cp.Constant(self._S)
        R_const = cp.Constant(R)

        n, m = self._S.shape
        W = cp.Variable((m,), complex=True)
        T = cp.Variable((m, m), hermitian=True)
        constraints = []
        for i in range(m):
            for j in range(i, m):
                k = abs(i - j)
                if k < m:
                    constraints.append(T[i, j] == W[k])
                    if i != j:
                        constraints.append(T[j, i] == cp.conj(W[k]))

        error = cp.multiply(M_const, T) - S_const.T @ R_const @ S_const.conj()
        objective = cp.Minimize(cp.norm(error, 'fro') ** 2 + self._zeta * cp.real(cp.trace(T)))
        constraints.append(T >> 0)

        prob = cp.Problem(objective, constraints)
        prob.solve(solver=self._solver, verbose=False)
        if prob.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            print(f"CVXPY Solver Error: {prob.status}")
            return None
        return T.value


class StructCovMLEEstimator(CovarianceReconstructionBase):
    """Creates a source location estimator based on Structured Covariance 
    Maximum Likelihood Estimation (StructCovMLE) algorithm.
    
    The StructCovMLE algorithm recovers the Toeplitz covariance matrix of a ULA 
    from sparse array data using an iterative optimization approach.
    
    Args:
        array (~doatools.model.arrays.ArrayDesign): Array design.
        wavelength (float): Wavelength of the carrier wave.
        search_grid (~doatools.estimation.grid.SearchGrid, optional): The search grid 
            used to locate the sources. Defaults to None. Not needed for 
            gridless DOA estimators like RootMUSIC1D and ESPRIT.
        epsilon (float, optional): Relative change threshold for convergence.
                                   Defaults to 1e-4.
        max_iter (int, optional): Maximum number of iterations. Defaults to 10.
        lambda_noise (float, optional): Noise variance for noise-aware variant.
        doa_estimator: DOA estimator instance. Defaults to None (RootMUSIC1D).
        **kwargs: Other keyword arguments.
            
    References:
        [1] R. R. Pote and B. D. Rao, "Maximum Likelihood-Based Gridless DoA Estimation Using Structured Covariance Matrix Recovery and SBL With Grid Refinement," *IEEE Transactions on Signal Processing*, vol. 71, pp. 802-815, 2023.
    """

    def __init__(self, array, wavelength, epsilon: float = 1e-4,
                 max_iter: int = 10, lambda_noise: float = None, doa_estimator=None, search_grid=None, **kwargs):
        super().__init__(array, wavelength, doa_estimator, search_grid, **kwargs)
        self._epsilon = epsilon
        self._max_iter = max_iter
        self._lambda_noise = lambda_noise

    def reconstruct(self, R, Y=None, **kwargs):
        """Reconstructs the augmented covariance matrix using StructCovMLE algorithm.
        
        Args:
            R (~numpy.ndarray): Sample covariance matrix of the sparse array.
            
        Returns:
            ~numpy.ndarray: Augmented covariance matrix.
        """
        ensure_covariance_size(R, self._array)
        n, m = self._S.shape
        V = np.eye(m, dtype=np.complex128)
        R_const = cp.Constant(R)
        S_const = cp.Constant(self._S)
        for iter_count in range(self._max_iter):
            V_prev = V.copy()
            Vs = self._S @ V @ self._S.conj().T
            if self._lambda_noise is not None:
                Vs += self._lambda_noise * np.eye(n)
            Vs_inv = inv(Vs)
            Vs_inv = cp.Constant((Vs_inv + Vs_inv.conj().T) / 2)
            T = cp.Variable((m, m), hermitian=True)
            X = cp.Variable((n, n), hermitian=True)
            toeplitz_constraints = []
            for i in range(1, m):
                for j in range(1, m):
                    toeplitz_constraints.append(T[i, j] == T[i - 1, j - 1])
            I_n = cp.Constant(np.eye(n))
            Z   = cp.Constant(np.zeros((n, m), dtype=np.complex128))
            lmi_matrix = cp.bmat([
                [X, I_n, Z],
                [I_n, S_const @ T @ S_const.conj().T, Z],
                [Z.conj().T, Z.conj().T, T]
            ])
            objective = cp.Minimize(
                cp.real(cp.trace(Vs_inv @ S_const @ T @ S_const.conj().T)) +
                cp.real(cp.trace(X @ R_const))
            )
            constraints = [lmi_matrix >> 0] + toeplitz_constraints
            problem = cp.Problem(objective, constraints)
            problem.solve(solver=self._solver, verbose=False)
            if problem.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                return None
            V = T.value
            relative_change = np.linalg.norm(V - V_prev, 'fro') / np.linalg.norm(V_prev, 'fro')
            if relative_change < self._epsilon:
                break
        V = (V + V.conj().T) / 2
        return V


class WassersteinEstimator(CovarianceReconstructionBase):
    """Creates a source location estimator based on Wasserstein algorithm.
    
    The Wasserstein algorithm estimates the Toeplitz covariance matrix of a 
    Uniform Linear Array (ULA) from the sample covariance matrix of a sparse 
    linear array using SDP.
    
    Args:
        array (~doatools.model.arrays.ArrayDesign): Array design.
        wavelength (float): Wavelength of the carrier wave.
        search_grid (~doatools.estimation.grid.SearchGrid, optional): The search grid 
            used to locate the sources. Defaults to None. Not needed for 
            gridless DOA estimators like RootMUSIC1D and ESPRIT.
        use_gradient (bool, optional): Whether to use the gradient-based 
                                      implementation. Defaults to False.
        mu (float, optional): log-barrier coefficient for gradient-based 
                            implementation. Defaults to 1e-2.
        lr (float, optional): Learning rate for gradient-based implementation.
                            Defaults to 1e-1.
        max_iter (int, optional): Maximum number of iterations for 
                                 gradient-based implementation. Defaults to 500.
        tol (float, optional): Convergence threshold for gradient-based 
                              implementation. Defaults to 1e-7.
        verbose (bool, optional): Whether to print verbose output for 
                                gradient-based implementation. Defaults to False.
        doa_estimator: DOA estimator instance. Defaults to None (RootMUSIC1D).
        **kwargs: Other keyword arguments.
            
    References:
        [1] M. Wang, Z. Zhang, and A. Nehorai, "Grid-Less DOA Estimation Using Sparse Linear Arrays Based on Wasserstein Distance," *IEEE Signal Processing Letters*, vol. 26, no. 6, pp. 838-842, June 2019.
    """

    def __init__(self, array, wavelength, use_gradient: bool = False,
                 mu: float = 1e-2, lr: float = 1e-1, max_iter: int = 500,
                 tol: float = 1e-7, verbose: bool = False, doa_estimator=None, search_grid=None, **kwargs):
        super().__init__(array, wavelength, doa_estimator, search_grid, **kwargs)
        self._use_gradient = use_gradient
        self._mu = mu
        self._lr = lr
        self._max_iter = max_iter
        self._tol = tol
        self._verbose = verbose

    def reconstruct(self, R, Y=None, **kwargs):
        """Reconstructs the augmented covariance matrix using Wasserstein algorithm.
        
        Args:
            R (~numpy.ndarray): Sample covariance matrix of the sparse array.
            
        Returns:
            ~numpy.ndarray: Augmented covariance matrix.
        """
        ensure_covariance_size(R, self._array)

        n, m = self._S.shape

        if self._use_gradient:
            def toeplitz_from_params(c):
                """
                利用参数向量 c 构建 Hermitian Toeplitz 矩阵
                c: [c_0, c_1, ..., c_{N-1}], 长度 N
                返回 N x N 锥形 Toeplitz 矩阵, 满足 R[i,j]=c[|i-j|]
                """
                N = len(c)
                T = np.zeros((N, N), dtype=np.complex128)
                for i in range(N):
                    for j in range(N):
                        T[i, j] = c[abs(i - j)]
                return (T + T.conj().T) / 2

            R_hat = (R + R.conj().T) / 2
            c = np.zeros(m, dtype=np.complex128)
            c[0] = np.real(np.trace(R_hat)) / m
            eigvals, eigvecs = np.linalg.eigh(R_hat)
            Rhat_sqrt = eigvecs @ np.diag(np.sqrt(np.maximum(eigvals, 0))) @ eigvecs.conj().T
            last_loss = np.inf
            for it in range(self._max_iter):
                R0 = toeplitz_from_params(c)
                RHS = self._S @ R0 @ self._S.conj().T
                try:
                    U, s, _ = np.linalg.svd(Rhat_sqrt @ RHS @ Rhat_sqrt)
                    sqrtA = U @ np.diag(np.sqrt(np.maximum(s, 0))) @ U.conj().T
                except Exception as e:
                    if self._verbose:
                        print(f"SVD/sqrt failed: {e}")
                    sqrtA = np.zeros_like(RHS)
                loss = np.real(np.trace(R_hat + RHS - 2 * sqrtA))
                sign, logdetR0 = np.linalg.slogdet(R0)
                if sign > 0:
                    loss -= self._mu * logdetR0
                else:
                    loss += np.inf  # Infeasible
                if abs(last_loss - loss) / (abs(loss) + 1e-9) < self._tol:
                    if self._verbose:
                        print(f"Iter {it} converge.")
                    break
                last_loss = loss

                grad = np.zeros_like(c, dtype=np.complex128)
                for k in range(m):
                    Q = np.zeros((m, m), dtype=np.complex128)
                    for i in range(m):
                        for j in range(m):
                            if abs(i - j) == k:
                                Q[i, j] = 1.0
                    Q = (Q + Q.conj().T) / 2
                    dRHS = self._S @ Q @ self._S.conj().T
                    try:
                        V = sqrtA
                        EV, s, _ = np.linalg.svd(V)
                        A = EV.conj().T @ (Rhat_sqrt @ dRHS @ Rhat_sqrt) @ EV
                        D = np.add.outer(s, s)
                        X = A / (D + 1e-8)
                        dtr_sqrtA = np.real(np.trace(X))
                    except Exception as e:
                        dtr_sqrtA = 0.0
                    grad[k] = np.real(np.trace(dRHS)) - 2 * dtr_sqrtA
                    if sign > 0:
                        grad[k] -= self._mu * np.real(np.trace(np.linalg.inv(R0) @ Q))
                    else:
                        grad[k] = 0.0
                c = c - self._lr * grad
                c[0] = np.real(c[0])
                if self._verbose and (it % 20 == 0 or it == self._max_iter - 1):
                    print(f"It={it}, Loss={loss:.5g}")
            R0 = toeplitz_from_params(c)
            return (R0 + R0.conj().T) / 2
        else:
            S_const  = cp.Constant(self._S)
            R_hat = (R + R.conj().T) / 2
            R_const  = cp.Constant(R_hat)
            R0 = cp.Variable((m, m), hermitian=True)
            V = cp.Variable((n, n), complex=True)

            objective = cp.Minimize(cp.real(cp.trace(R_const + S_const @ R0 @ S_const.conj().T - V - V.conj().T)))
            constraints = []
            for i in range(m):
                for j in range(m):
                    if i > 0 and j > 0:
                        constraints.append(R0[i, j] == R0[i - 1, j - 1])
            lmi_matrix = cp.bmat([
                [S_const @ R0 @ S_const.conj().T, V],
                [V.conj().T, R_const]
            ])
            constraints.append(lmi_matrix >> 0)
            constraints.append(R0 >> 0)
            problem = cp.Problem(objective, constraints)

            problem.solve(solver=self._solver, verbose=False)
            if problem.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                return None
            else:
                result = (R0.value + R0.value.conj().T) / 2
                result = toeplitz(result[:, 0])
                result = (result + result.conj().T) / 2
                return result

from scipy.optimize import least_squares
from scipy.signal import find_peaks

class NewtonianIdentitiesBeamformingEstimator(CovarianceReconstructionBase):
    """基于牛顿恒等式 + 确定性粗波束形成，联合估计信号功率 σ_s² 和初值 e"""

    def __init__(self, array, wavelength, sigma_s2: float = 1.0,
                 doa_estimator=None, search_grid=None,
                 beamforming_grid_size=1800, **kwargs):
        super().__init__(array, wavelength, doa_estimator, search_grid, **kwargs)
        self._sigma_s2 = sigma_s2          # 仅作为 fallback，实际会被估计覆盖
        self._bf_grid_size = beamforming_grid_size
        self._d_lambda = 0.5
        self.estimated_sigma_s2 = None     # 新增：运行后可访问真实估计值

    @staticmethod
    def _compute_elementary_symmetric_polynomials(p: np.ndarray, k: int) -> np.ndarray:
        e = np.zeros(k + 1, dtype=complex)
        e[0] = 1
        for n in range(1, k + 1):
            sum_term = 0
            for j in range(1, n + 1):
                if j - 1 < len(p):
                    sum_term += (-1) ** (j - 1) * e[n - j] * p[j - 1]
            e[n] = sum_term / n
        return e[1:]

    @staticmethod
    def _compute_power_sums_from_e(e: np.ndarray, k: int, m: int) -> np.ndarray:
        p_array = np.zeros(m, dtype=complex)
        for n in range(1, m + 1):
            if n <= k:
                p_n = (-1) ** (n - 1) * n * e[n - 1]
                for j in range(1, n):
                    p_n += (-1) ** (j - 1) * e[j - 1] * p_array[n - j - 1]
            else:
                p_n = 0
                for j in range(1, k + 1):
                    p_n += (-1) ** (j - 1) * e[j - 1] * p_array[n - j - 1]
            p_array[n - 1] = p_n
        return p_array

    def _angles_to_e(self, angles, k):
        zs = np.exp(1j * 2 * np.pi * self._d_lambda * np.sin(angles))
        coeffs = np.poly(zs)
        e_vec = np.zeros(k, dtype=complex)
        for n in range(1, k + 1):
            e_vec[n-1] = coeffs[n] * ((-1)**n)
        return e_vec

    def _get_initial_guess(self, k, known_p, known_indices):
        """波束形成 + 线性LS，同时给出 e_init 和 sigma_init（关键改进）"""
        grid = np.linspace(-np.pi/2, np.pi/2, self._bf_grid_size)
        mu = 2 * np.pi * self._d_lambda * np.sin(grid)
        exponents = np.exp(-1j * np.outer(known_indices, mu))
        spectrum = np.real(known_p @ exponents)

        # 改进寻峰：避免相邻重复峰
        peaks, _ = find_peaks(spectrum, distance=max(3, len(grid)//100))
        if len(peaks) >= k:
            top_idx = np.argsort(spectrum[peaks])[-k:]
            selected_angles = grid[peaks[top_idx]]
        else:
            selected_angles = grid[np.argsort(spectrum)[-k:]]

        e_init = self._angles_to_e(selected_angles, k)

        # 用初始 e 做线性LS求 sigma_init（极快且准）
        max_known = int(np.max(known_indices))
        p_init = self._compute_power_sums_from_e(e_init, k, max_known)
        p_known_init = p_init[known_indices - 1]
        denom = np.sum(np.abs(p_known_init)**2)
        if denom > 1e-12:
            num = np.dot(known_p, np.conj(p_known_init))
            sigma_init = max(np.real(num / denom), 1e-6)
        else:
            sigma_init = 1.0
        return e_init, sigma_init

    def _compute_missing_R(self, k: int, m: int, known_p: np.ndarray, known_indices: np.ndarray):
        """联合优化 e 和 σ_s²，返回 (补全的R, estimated_sigma)"""
        max_known = int(np.max(known_indices))

        # 1. 优质初值（波束形成 + 线性LS）
        e_init, sigma_init = self._get_initial_guess(k, known_p, known_indices)

        # 2. 优化变量： [Re(e), Im(e), sigma]
        x0 = np.zeros(2 * k + 1)
        x0[:k] = np.real(e_init)
        x0[k:2*k] = np.imag(e_init)
        x0[-1] = sigma_init

        def residual_function(x):
            e = x[:k] + 1j * x[k:2*k]
            sigma = max(x[-1], 1e-8)                     # 安全保护
            p_array = self._compute_power_sums_from_e(e, k, max_known)
            pred = sigma * p_array[known_indices - 1]
            diff = pred - known_p
            return np.concatenate([np.real(diff), np.imag(diff)])

        # 3. 优化（trf 支持 bounds）
        lb = np.full(2*k + 1, -np.inf)
        lb[-1] = 1e-8
        ub = np.full(2*k + 1, np.inf)
        res = least_squares(residual_function, x0, bounds=(lb, ub),
                            method='trf', ftol=1e-12, xtol=1e-12,
                            gtol=1e-12, max_nfev=2000)

        # 4. 提取结果
        x_opt = res.x
        e_opt = x_opt[:k] + 1j * x_opt[k:2*k]
        sigma_est = max(x_opt[-1], 1e-8)

        p_full = self._compute_power_sums_from_e(e_opt, k, m)
        return sigma_est * p_full, sigma_est

    def reconstruct(self, R, **kwargs):
        return self._coarray_builder.transform(R, 'da')

    def estimate(self, R, k, **kwargs):
        _, m = self._S.shape
        scm_vector = self.reconstruct(R)[:, 0]
        non_zero_mask = scm_vector[1:] != 0.0
        known_indices = np.where(non_zero_mask)[0] + 1
        known_values = scm_vector[known_indices]

        try:
            all_R_values, sigma_est = self._compute_missing_R(
                k, m - 1, known_values, known_indices)
            scm_vector[1:m] = all_R_values
            self.estimated_sigma_s2 = sigma_est
        except Exception as e:
            print(f"niBFAlgorithmFails: {e}")
            self.estimated_sigma_s2 = self._sigma_s2

        Ra = toeplitz(scm_vector)
        Ra = (Ra + Ra.conj().T) / 2
        return self._doa_estimate(Ra, k, **kwargs)

