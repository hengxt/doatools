from abc import ABC, abstractmethod
import numpy as np
from scipy.linalg import inv, sqrtm, toeplitz

import cvxpy as cp

from .core import SpectrumBasedEstimatorBase, ensure_covariance_size
from .coarray import CoarrayACMBuilder1D


class CovarianceReconstructionBase(ABC):
    """Base class for covariance matrix reconstruction algorithms.
    
    Covariance matrix reconstruction algorithms aim to transform the sample 
    covariance matrix of a sparse array into an augmented covariance matrix 
    that corresponds to a larger virtual uniform linear array, typically 
    based on the difference coarray concept.
    """
    
    @abstractmethod
    def reconstruct(self, R):
        """Reconstructs the augmented covariance matrix.
        
        Args:
            R (~numpy.ndarray): Sample covariance matrix of the sparse array.
            
        Returns:
            ~numpy.ndarray: Augmented covariance matrix.
        """
        pass
    
    @property
    @abstractmethod
    def virtual_array(self):
        """Retrieves the corresponding virtual uniform linear array.
        
        Returns:
            ~doatools.model.arrays.UniformLinearArray: Virtual uniform linear array.
        """
        pass


class SPAEstimator(SpectrumBasedEstimatorBase):
    """Creates a source location estimator based on Spatial-smoothing based 
    Augmentation (SPA) algorithm.
    
    The SPA algorithm reconstructs a large covariance matrix corresponding to 
    the difference coarray uniform linear array using semi-definite programming 
    (SDP) approach, and then applies subspace-based methods (such as MUSIC) 
    for DOA estimation.
    
    Args:
        array (~doatools.model.arrays.ArrayDesign): Array design.
        wavelength (float): Wavelength of the carrier wave.
        search_grid (~doatools.estimation.grid.SearchGrid): The search grid 
            used to locate the sources.
        lambda_noise (float, optional): The noise variance, used for the 
                                        SPA_noisevar variant. Defaults to 0
                                        (standard SPA).
        **kwargs: Other keyword arguments supported by 
            :class:`~doatools.estimation.core.SpectrumBasedEstimatorBase`.
            
    References:
        [1] X. Yuan and A. Nehorai, "Sparse Arrays for DOA Estimation: 
        From Coarray Perspective," IEEE Transactions on Signal Processing, 
        vol. 66, no. 4, pp. 939-953, Feb. 2018.
    """
    
    def __init__(self, array, wavelength, search_grid, lambda_noise: float = 0, **kwargs):
        super().__init__(array, wavelength, search_grid, **kwargs)
        # Create a coarray ACM builder for reference
        self._coarray_builder = CoarrayACMBuilder1D(array)
        self._virtual_ula = self._coarray_builder.get_virtual_ula()
        self._lambda_noise = lambda_noise
        # Create selection matrix S based on the coarray mapping
        # This is a placeholder - actual implementation should compute S based on the array
        self._S = np.eye(array.size, self._virtual_ula.size, dtype=np.complex128)
    
    def reconstruct(self, R):
        """Reconstructs the augmented covariance matrix using SPA algorithm.
        
        Args:
            R (~numpy.ndarray): Sample covariance matrix of the sparse array.
            
        Returns:
            ~numpy.ndarray: Augmented covariance matrix.
        """
        ensure_covariance_size(R, self._array)
        # Use SPA algorithm to reconstruct the covariance matrix
        n, m = self._S.shape
        
        # Ensure R_hat is hermitian
        R_hat = (R + R.T.conj()) / 2

        # Calculate R_hat inverse and square root
        try:
            R_hat_inv = inv(R_hat)
            R_sqrt = sqrtm(R_hat)
        except np.linalg.LinAlgError:
            print("Error: R_hat is singular or not positive semidefinite.")
            return None

        # Ensure intermediate matrices are hermitian (due to potential numerical issues)
        R_hat_inv = (R_hat_inv + R_hat_inv.T.conj()) / 2
        R_sqrt = (R_sqrt + R_sqrt.T.conj()) / 2

        # Define variables
        # T is m x m hermitian toeplitz. Define as a general hermitian matrix first.
        T = cp.Variable((m, m), hermitian=True)

        # X is n x n hermitian complex auxiliary variable
        X = cp.Variable((n, n), hermitian=True)

        # Objective function: minimize trace(X) + trace(inv(R_hat) * S * T * S')
        objective = cp.Minimize(cp.real(cp.trace(X) + cp.trace(R_hat_inv @ self._S @ T @ self._S.T.conj())))

        # LMI constraint
        Z = np.zeros((n, m), dtype=complex)
        I_n = np.eye(n)

        if self._lambda_noise > 0:
            lmi_matrix = cp.bmat([
                [X, R_sqrt, Z],
                [R_sqrt.T.conj(), self._S @ T @ self._S.T.conj() + self._lambda_noise * I_n, Z],
                [Z.T.conj(), Z.T.conj(), T]
            ])
        else:
            lmi_matrix = cp.bmat([
                [X, R_sqrt, Z],
                [R_sqrt.T.conj(), self._S @ T @ self._S.T.conj(), Z],
                [Z.T.conj(), Z.T.conj(), T]
            ])

        # Constraints
        constraints = [lmi_matrix >> 0]  # SDP constraint

        # Add Toeplitz constraints to T: T_ij = T_{i+1, j+1} for all valid i, j
        for i in range(m - 1):
            for j in range(m - 1):
                constraints.append(T[i, j] == T[i + 1, j + 1])

        # Solve the problem
        problem = cp.Problem(objective, constraints)

        # Solve the problem. SCS is a common solver for complex SDPs.
        try:
            # Attempt solving with MOSEK
            problem.solve(solver=cp.MOSEK)

        except cp.SolverError as e:
            print(f"SPA: CVXPY Solver Error: {e}")
            return None
        except Exception as e:
            print(f"An unexpected error occurred during problem solving: {e}")
            return None

        # Check the problem status
        if problem.status != cp.OPTIMAL:
            print(f"Problem did not solve to optimality. Status: {problem.status}")
            return None

        # Return the optimal Toeplitz matrix T
        estimated_T = T.value
        if estimated_T is not None:
            first_row = estimated_T[:, 0]
            estimated_T = toeplitz(first_row)
            # Ensure hermitian property in the final output
            estimated_T = (estimated_T + estimated_T.T.conj()) / 2

        return estimated_T
    
    def estimate(self, R, k, **kwargs):
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
              same type as the one used in the search grid, representing the 
              estimated source locations. Will be ``None`` if resolved is 
              ``False``.
            * spectrum (:class:`~numpy.ndarray`): An numpy array of the same 
              shape of the specified search grid, consisting of values evaluated 
              at the grid points. Only present if ``return_spectrum`` is 
              ``True``.
        """
        ensure_covariance_size(R, self._array)
        # Reconstruct the covariance matrix
        Ra = self.reconstruct(R)
        if Ra is None:
            # Reconstruction failed
            if 'return_spectrum' in kwargs and kwargs['return_spectrum']:
                return False, None, None
            else:
                return False, None
        # Use the virtual ULA for DOA estimation
        return super().estimate(Ra, k, **kwargs)
    
    @property
    def virtual_array(self):
        """Retrieves the corresponding virtual uniform linear array.
        
        Returns:
            ~doatools.model.arrays.UniformLinearArray: Virtual uniform linear array.
        """
        return self._virtual_ula


class ANMEstimator(SpectrumBasedEstimatorBase):
    """Creates a source location estimator based on Augmentation via 
    Nuclear-norm Minimization (ANM) algorithm.
    
    The ANM algorithm reconstructs the augmented covariance matrix by solving 
    a nuclear-norm minimization problem, which exploits the low-rank property 
    of the covariance matrix.
    
    Args:
        array (~doatools.model.arrays.ArrayDesign): Array design.
        wavelength (float): Wavelength of the carrier wave.
        search_grid (~doatools.estimation.grid.SearchGrid): The search grid 
            used to locate the sources.
        zeta (float, optional): Regularization parameter. Defaults to 0.0.
        max_workers (int, optional): Maximum number of workers for parallel 
                                      processing. Defaults to 1.
        **kwargs: Other keyword arguments supported by 
            :class:`~doatools.estimation.core.SpectrumBasedEstimatorBase`.
            
    References:
        [1] Q. Wan, et al., "Augmented Covariance Matrix Reconstruction for 
        Sparse Arrays via Nuclear-norm Minimization," IEEE Transactions on 
        Signal Processing, vol. 67, no. 11, pp. 2896-2910, Jun. 2019.
    """
    
    def __init__(self, array, wavelength, search_grid, zeta: float = 0.0, max_workers: int = 1, **kwargs):
        super().__init__(array, wavelength, search_grid, **kwargs)
        # Create a coarray ACM builder for reference
        self._coarray_builder = CoarrayACMBuilder1D(array)
        self._virtual_ula = self._coarray_builder.get_virtual_ula()
        self._zeta = zeta
        self._max_workers = max_workers
        # Create mask matrix (identity matrix for full array)
        self._mask_matrix = np.eye(array.size, dtype=np.complex128)
    
    def reconstruct(self, R):
        """Reconstructs the augmented covariance matrix using ANM algorithm.
        
        Args:
            R (~numpy.ndarray): Sample covariance matrix of the sparse array.
            
        Returns:
            ~numpy.ndarray: Augmented covariance matrix.
        """
        ensure_covariance_size(R, self._array)
        # Use ANM algorithm to reconstruct the covariance matrix
        N = R.shape[0]
        
        def _sparse_to_full_array_interpolation_single(cov_matrix, mask_matrix__, zeta_):
            # 优化变量 - 修正W的定义
            W = cp.Variable((N,), complex=True)  # 长度为N的复数向量

            # Hermitian Toeplitz矩阵
            T = cp.Variable((N, N), hermitian=True)

            # 正确构建Toeplitz约束
            constraints = []
            for i in range(N):
                for j in range(i, N):
                    k = abs(i - j)  # Toeplitz索引
                    if k < N:  # 确保索引在范围内
                        constraints.append(T[i, j] == W[k])
                        if i != j:
                            constraints.append(T[j, i] == cp.conj(W[k]))

            # 定义优化问题
            error = cp.multiply(mask_matrix__, T) - cov_matrix
            objective = cp.Minimize(cp.norm(error, 'fro') ** 2 + zeta_ * cp.real(cp.trace(T)))
            constraints.append(T >> 0)  # 半正定约束

            # 求解 - 优先使用SCS，避免MOSEK问题
            prob = cp.Problem(objective, constraints)

            try:
                # 优先使用SCS求解器，避免MOSEK兼容性问题
                prob.solve(solver=cp.MOSEK, verbose=False)
                # 如果SCS失败，尝试其他求解器
                if prob.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                    try:
                        prob.solve(solver=cp.ECOS, verbose=False)
                    except:
                        # 最后尝试SCS
                        try:
                            prob.solve(solver=cp.SCS, verbose=False, eps=1e-4, max_iters=5000)
                        except:
                            pass

            except Exception as e:
                print(f"ANM:MOSEK算法求解器错误: {str(e)}， 换用ECOS求解器")
                try:
                    prob.solve(solver=cp.ECOS, verbose=False)
                    return T.value
                except:
                    print(f"ECOS求解器错误: {str(e)}, 换SCS求解器")
                    # 最后尝试SCS
                    try:
                        prob.solve(solver=cp.SCS, verbose=False, eps=1e-4, max_iters=5000)
                        return T.value
                    except:
                        print("SCS求解器错误")
                        pass
                # 返回原始矩阵或简单估计
                return cov_matrix

            if prob.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                print(f"优化失败，状态: {prob.status}")
                return cov_matrix

            return T.value
        
        # 如果输入是 3D 数组（batchsize, M, M），则对每个样本并行处理
        if R.ndim == 3:
            batchsize, M, _ = R.shape
            results = np.zeros((batchsize, M, M), dtype=complex)

            # 使用线程池并行处理
            from concurrent.futures import ThreadPoolExecutor, as_completed
            with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
                futures = []
                for i in range(batchsize):
                    futures.append(
                        executor.submit(_sparse_to_full_array_interpolation_single,
                                        R[i], self._mask_matrix, self._zeta)
                    )

                # 收集结果
                for i, future in enumerate(as_completed(futures)):
                    try:
                        results[i] = future.result()
                    except Exception as e:
                        print(f"处理样本 {i} 时出错: {str(e)}")
                        results[i] = R[i]  # 出错时返回原始矩阵

            return results
        else:
            # 如果输入是 2D 数组（M, M），直接处理
            return _sparse_to_full_array_interpolation_single(R, self._mask_matrix, self._zeta)
    
    def estimate(self, R, k, **kwargs):
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
              same type as the one used in the search grid, representing the 
              estimated source locations. Will be ``None`` if resolved is 
              ``False``.
            * spectrum (:class:`~numpy.ndarray`): An numpy array of the same 
              shape of the specified search grid, consisting of values evaluated 
              at the grid points. Only present if ``return_spectrum`` is 
              ``True``.
        """
        ensure_covariance_size(R, self._array)
        # Reconstruct the covariance matrix
        Ra = self.reconstruct(R)
        if Ra is None:
            # Reconstruction failed
            if 'return_spectrum' in kwargs and kwargs['return_spectrum']:
                return False, None, None
            else:
                return False, None
        # Use the virtual ULA for DOA estimation
        return super().estimate(Ra, k, **kwargs)
    
    @property
    def virtual_array(self):
        """Retrieves the corresponding virtual uniform linear array.
        
        Returns:
            ~doatools.model.arrays.UniformLinearArray: Virtual uniform linear array.
        """
        return self._virtual_ula


class StructCovMLEEstimator(SpectrumBasedEstimatorBase):
    """Creates a source location estimator based on Structured Covariance 
    Maximum Likelihood Estimation (StructCovMLE) algorithm.
    
    The StructCovMLE algorithm recovers the Toeplitz covariance matrix of a ULA 
    from sparse array data using an iterative optimization approach.
    
    Args:
        array (~doatools.model.arrays.ArrayDesign): Array design.
        wavelength (float): Wavelength of the carrier wave.
        search_grid (~doatools.estimation.grid.SearchGrid): The search grid 
            used to locate the sources.
        epsilon (float, optional): Relative change threshold for convergence.
                                   Defaults to 1e-4.
        max_iter (int, optional): Maximum number of iterations. Defaults to 10.
        lambda_noise (float, optional): Noise variance for noise-aware variant.
        **kwargs: Other keyword arguments supported by 
            :class:`~doatools.estimation.core.SpectrumBasedEstimatorBase`.
            
    References:
        [1] R. R. Pote and B. D. Rao, "Maximum Likelihood-Based Gridless DoA 
        Estimation Using Structured Covariance Matrix Recovery and SBL With Grid 
        Refinement," in IEEE Transactions on Signal Processing, vol. 71, 
        pp. 802-815, 2023.
    """
    
    def __init__(self, array, wavelength, search_grid, epsilon: float=1e-4, 
                 max_iter: int=10, lambda_noise: float = None, **kwargs):
        super().__init__(array, wavelength, search_grid, **kwargs)
        # Create a coarray ACM builder for reference
        self._coarray_builder = CoarrayACMBuilder1D(array)
        self._virtual_ula = self._coarray_builder.get_virtual_ula()
        self._epsilon = epsilon
        self._max_iter = max_iter
        self._lambda_noise = lambda_noise
        # Create selection matrix S based on the coarray mapping
        # This is a placeholder - actual implementation should compute S based on the array
        self._S = np.eye(array.size, self._virtual_ula.size, dtype=np.complex128)
    
    def reconstruct(self, R):
        """Reconstructs the augmented covariance matrix using StructCovMLE algorithm.
        
        Args:
            R (~numpy.ndarray): Sample covariance matrix of the sparse array.
            
        Returns:
            ~numpy.ndarray: Augmented covariance matrix.
        """
        ensure_covariance_size(R, self._array)
        # Use StructCovMLE algorithm to reconstruct the covariance matrix
        n, m = self._S.shape
        
        # Validate input covariance matrix
        if not (np.allclose(R, R.conj().T)):
            raise ValueError("Input matrix R_hat must be Hermitian")

        # Initialize with identity matrix
        V = np.eye(m, dtype=np.complex128)

        for iter_count in range(self._max_iter):
            V_prev = V.copy()

            # Compute inverse term with numerical stability
            Vs = self._S @ V @ self._S.conj().T
            if self._lambda_noise is not None:
                Vs += self._lambda_noise * np.eye(n)
            Vs_inv = inv(Vs)
            Vs_inv = (Vs_inv + Vs_inv.conj().T) / 2  # Ensure Hermitian

            # Define optimization variables
            T = cp.Variable((m, m), hermitian=True)
            X = cp.Variable((n, n), hermitian=True)

            # Correct Toeplitz constraints
            toeplitz_constraints = []
            for i in range(1, m):
                for j in range(1, m):
                    toeplitz_constraints.append(T[i, j] == T[i-1, j-1])

            # Build LMI matrix
            Z = np.zeros((n, m), dtype=np.complex128)
            I_n = np.eye(n)
            lmi_matrix = cp.bmat([
                [X, I_n, Z],
                [I_n, self._S @ T @ self._S.conj().T, Z],
                [Z.conj().T, Z.conj().T, T]
            ])

            # Objective function
            objective = cp.Minimize(
                cp.real(cp.trace(Vs_inv @ self._S @ T @ self._S.conj().T)) +
                cp.real(cp.trace(X @ R))
            )

            # Constraints
            constraints = [lmi_matrix >> 0] + toeplitz_constraints

            # Solve optimization problem
            try:
                problem = cp.Problem(objective, constraints)
                problem.solve(solver=cp.MOSEK, verbose=False)

                if problem.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                    raise RuntimeError(f"Solver failed with status: {problem.status}")

                # Update V with solution
                V = T.value

                # Check convergence
                relative_change = np.linalg.norm(V - V_prev, 'fro') / np.linalg.norm(V_prev, 'fro')
                if relative_change < self._epsilon:
                    break

            except cp.SolverError as e:
                print(f"STRUCT: CVXPY solver error: {e}")
                return None
            except Exception as e:
                print(f"Unexpected error in optimization: {e}")
                return None

        # Return the solution directly (should already be Toeplitz)
        if V is not None:
            # Ensure Hermitian property due to numerical errors
            V = (V + V.conj().T) / 2

        return V
    
    def estimate(self, R, k, **kwargs):
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
              same type as the one used in the search grid, representing the 
              estimated source locations. Will be ``None`` if resolved is 
              ``False``.
            * spectrum (:class:`~numpy.ndarray`): An numpy array of the same 
              shape of the specified search grid, consisting of values evaluated 
              at the grid points. Only present if ``return_spectrum`` is 
              ``True``.
        """
        ensure_covariance_size(R, self._array)
        # Reconstruct the covariance matrix
        Ra = self.reconstruct(R)
        if Ra is None:
            # Reconstruction failed
            if 'return_spectrum' in kwargs and kwargs['return_spectrum']:
                return False, None, None
            else:
                return False, None
        # Use the virtual ULA for DOA estimation
        return super().estimate(Ra, k, **kwargs)
    
    @property
    def virtual_array(self):
        """Retrieves the corresponding virtual uniform linear array.
        
        Returns:
            ~doatools.model.arrays.UniformLinearArray: Virtual uniform linear array.
        """
        return self._virtual_ula


class WassersteinEstimator(SpectrumBasedEstimatorBase):
    """Creates a source location estimator based on Wasserstein algorithm.
    
    The Wasserstein algorithm estimates the Toeplitz covariance matrix of a 
    Uniform Linear Array (ULA) from the sample covariance matrix of a sparse 
    linear array using SDP.
    
    Args:
        array (~doatools.model.arrays.ArrayDesign): Array design.
        wavelength (float): Wavelength of the carrier wave.
        search_grid (~doatools.estimation.grid.SearchGrid): The search grid 
            used to locate the sources.
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
        **kwargs: Other keyword arguments supported by 
            :class:`~doatools.estimation.core.SpectrumBasedEstimatorBase`.
            
    References:
        [1] M. Wang, Z. Zhang and A. Nehorai, "Grid-Less DOA Estimation Using 
        Sparse Linear Arrays Based on Wasserstein Distance," in IEEE Signal 
        Processing Letters, vol. 26, no. 6, pp. 838-842, June 2019.
    """
    
    def __init__(self, array, wavelength, search_grid, use_gradient: bool = False,
                 mu: float = 1e-2, lr: float = 1e-1, max_iter: int = 500,
                 tol: float = 1e-7, verbose: bool = False, **kwargs):
        super().__init__(array, wavelength, search_grid, **kwargs)
        # Create a coarray ACM builder for reference
        self._coarray_builder = CoarrayACMBuilder1D(array)
        self._virtual_ula = self._coarray_builder.get_virtual_ula()
        self._use_gradient = use_gradient
        self._mu = mu
        self._lr = lr
        self._max_iter = max_iter
        self._tol = tol
        self._verbose = verbose
        # Create selection matrix S based on the coarray mapping
        # This is a placeholder - actual implementation should compute S based on the array
        self._S = np.eye(array.size, self._virtual_ula.size, dtype=np.complex128)
    
    def reconstruct(self, R):
        """Reconstructs the augmented covariance matrix using Wasserstein algorithm.
        
        Args:
            R (~numpy.ndarray): Sample covariance matrix of the sparse array.
            
        Returns:
            ~numpy.ndarray: Augmented covariance matrix.
        """
        ensure_covariance_size(R, self._array)
        # Use Wasserstein algorithm to reconstruct the covariance matrix
        n, m = self._S.shape
        
        if self._use_gradient:
            # Gradient-based implementation
            # 确保 R_hat Hermitian
            R_hat = (R + R.conj().T)/2

            # 初始化参数 c（对角线，实数）
            c = np.zeros(m, dtype=np.complex128)
            c[0] = np.real(np.trace(R_hat))/(m)

            # 预处理 eig of R_hat
            eigvals, eigvecs = np.linalg.eigh(R_hat)
            Rhat_invsqrt = eigvecs @ np.diag(1/np.sqrt(np.maximum(eigvals,1e-8))) @ eigvecs.conj().T
            Rhat_sqrt = eigvecs @ np.diag(np.sqrt(np.maximum(eigvals,0))) @ eigvecs.conj().T

            last_loss = np.inf

            for it in range(self._max_iter):
                # 构造 Toeplitz R0
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
                            T[i, j] = c[abs(i-j)]
                    # 保证 Hermitian
                    return (T + T.conj().T)/2
                
                R0 = toeplitz_from_params(c)
                # S*R0*S^H
                RHS = self._S @ R0 @ self._S.conj().T
                # sqrt 矩阵
                try:
                    # eig分解，避免复杂度
                    U, s, _ = np.linalg.svd(Rhat_sqrt @ RHS @ Rhat_sqrt)
                    sqrtA = U @ np.diag(np.sqrt(np.maximum(s,0))) @ U.conj().T
                except Exception as e:
                    if self._verbose:
                        print(f"SVD/sqrt failed: {e}")
                    sqrtA = np.zeros_like(RHS)
                # 目标函数值
                loss = np.real(np.trace(R_hat + RHS - 2*sqrtA))
                # log-barrier
                sign, logdetR0 = np.linalg.slogdet(R0)
                if sign > 0:
                    loss -= self._mu*logdetR0
                else:
                    loss += np.inf # Infeasible

                # 收敛
                if abs(last_loss-loss)/(abs(loss)+1e-9)<self._tol:
                    if self._verbose:
                        print(f"Iter {it} converge.")
                    break
                last_loss = loss

                # 梯度计算
                grad = np.zeros_like(c,dtype=np.complex128)
                # g1: d tr(S R0 S^H)/dc_k
                for k in range(m):
                    # 构造 (k)阶 Toeplitz 基
                    Q = np.zeros((m,m),dtype=np.complex128)
                    for i in range(m):
                        for j in range(m):
                            if abs(i-j)==k:
                                Q[i,j] = 1.0
                    Q = (Q+Q.conj().T)/2
                    # dRHS/dc_k = S Q S^H
                    dRHS = self._S@Q@self._S.conj().T
                    # d sqrtA/dc_k: 利用论文公式
                    # (V dV/dc + dV/dc V) = Rhat_sqrt S Q S^H Rhat_sqrt
                    try:
                        V = sqrtA
                        EV, s, _ = np.linalg.svd(V)
                        # s 是奇异值的1D数组
                        ones = np.ones_like(s)
                        A = EV.conj().T @ (Rhat_sqrt @ dRHS @ Rhat_sqrt) @ EV
                        D = np.add.outer(s, s) # λ_i+λ_j (1D数组)
                        X = A/(D+1e-8)
                        dtr_sqrtA = np.real(np.trace(X))
                    except Exception as e:
                        dtr_sqrtA = 0.0
                    grad[k] = np.real(np.trace(dRHS)) - 2*dtr_sqrtA
                    # logdet barrier
                    if sign > 0:
                        grad[k] -= self._mu * np.real(np.trace(np.linalg.inv(R0)@Q))
                    else:
                        grad[k] = 0.0
                # 梯度下降
                c = c - self._lr * grad
                # 保持对称（主元一定实数）
                c[0] = np.real(c[0])
                if self._verbose and (it%20==0 or it==self._max_iter-1):
                    print(f"It={it}, Loss={loss:.5g}")
            # 输出最终 Toeplitz
            R0 = toeplitz_from_params(c)
            return (R0+R0.conj().T)/2
        else:
            # SDP-based implementation
            # Ensure R_hat is Hermitian
            R_hat = (R + R.conj().T) / 2

            # Define variables
            R0 = cp.Variable((m, m), hermitian=True)  # Estimated Toeplitz covariance matrix
            V = cp.Variable((n, n), complex=True)  # Auxiliary variable

            # Objective function: minimize trace(R_hat + S*R0*S' - V - V')
            objective = cp.Minimize(cp.real(cp.trace(R_hat + self._S @ R0 @ self._S.conj().T - V - V.conj().T)))

            # Add Toeplitz constraints to R0 (corrected)
            constraints = []
            for i in range(m):
                for j in range(m):
                    if i > 0 and j > 0:
                        constraints.append(R0[i, j] == R0[i-1, j-1])

            # LMI constraint: [S*R0*S' V; V' R_hat] >= 0
            lmi_matrix = cp.bmat([
                [self._S @ R0 @ self._S.conj().T, V],
                [V.conj().T, R_hat]
            ])
            constraints.append(lmi_matrix >> 0)

            # Non-negative constraint on R0
            constraints.append(R0 >> 0)

            # Solve the problem
            problem = cp.Problem(objective, constraints)

            try:
                problem.solve(solver=cp.MOSEK, verbose=False)
            except cp.SolverError as e:
                print(f"WDA: CVXPY Solver Error: {e}")
                return None
            except Exception as e:
                print(f"An unexpected error occurred during problem solving: {e}")
                return None

            # Check solution status
            if problem.status != cp.OPTIMAL:
                print(f"Problem did not solve to optimality. Status: {problem.status}")
                return None

            # Return the solution
            if R0.value is not None:
                # Ensure Hermitian property due to numerical errors
                result = (R0.value + R0.value.conj().T) / 2
                return result

            return None
    
    def estimate(self, R, k, **kwargs):
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
              same type as the one used in the search grid, representing the 
              estimated source locations. Will be ``None`` if resolved is 
              ``False``.
            * spectrum (:class:`~numpy.ndarray`): An numpy array of the same 
              shape of the specified search grid, consisting of values evaluated 
              at the grid points. Only present if ``return_spectrum`` is 
              ``True``.
        """
        ensure_covariance_size(R, self._array)
        # Reconstruct the covariance matrix
        Ra = self.reconstruct(R)
        if Ra is None:
            # Reconstruction failed
            if 'return_spectrum' in kwargs and kwargs['return_spectrum']:
                return False, None, None
            else:
                return False, None
        # Use the virtual ULA for DOA estimation
        return super().estimate(Ra, k, **kwargs)
    
    @property
    def virtual_array(self):
        """Retrieves the corresponding virtual uniform linear array.
        
        Returns:
            ~doatools.model.arrays.UniformLinearArray: Virtual uniform linear array.
        """
        return self._virtual_ula