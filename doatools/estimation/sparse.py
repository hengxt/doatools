import numpy as np
import warnings
try:
    import cvxpy as cvx
    cvx_available = True
except ImportError:
    warnings.warn('Cannot import cvxpr. Some sparse recovery based estimators will not be usable.')
    cvx_available = False

from .core import SpectrumBasedEstimatorBase, ensure_covariance_size
from ..utils.math import khatri_rao, vec


class L1RegularizedLeastSquaresProblem:
    r"""Creates a reusable :math:`l_1`-regularized least squares problem.

    Let :math:`\mathbf{A}` be an :math:`M \times L` real dictionary matrix,
    :math:`\mathbf{b}` be an :math:`M \times 1` observation vector,
    :math:`\mathbf{x}` be a :math:`K \times 1` sparse vector, :math:`l` be
    a nonnegative scalar. Let :math:`c` equal to 0 if :math:`\mathbf{x}`
    must be nonnegative and :math:`-\infty` if :math:`\mathbf{x}` can be any
    real number.
    
    The default formulation, named ``'penalizedl1'``, is given by

    .. math::

        \begin{aligned}
        \min_{\mathbf{x}}&
        \frac{1}{2} \| \mathbf{A}\mathbf{x} - \mathbf{b} \|_2^2 +
            l \| \mathbf{x} \|_1,\\
        \text{s.t. }& x \geq c
        \end{aligned}

    This formulation can be efficiently solved with QP or FISTA.

    One common variant, namely the ``'constraintedl1'`` formulation, is
    given by

    .. math::

        \begin{aligned}
        \min_{\mathbf{x}}& \| \mathbf{A}\mathbf{x} - \mathbf{b} \|_2^2,\\
        \text{s.t. }& \| \mathbf{x} \|_1 \leq l, \mathbf{x} \geq c.
        \end{aligned}

    This formulation can be efficiently solved with QP.

    A less common variant, namely the ``'constrainedl2'`` formulation, is
    given by

    .. math::

        \begin{aligned}
        \min_{\mathbf{x}}& \| \mathbf{x} \|_1,\\
        \text{s.t. }& \| \mathbf{A}\mathbf{x} - \mathbf{b} \|_2 \leq l,
            \mathbf{x} \geq c.
        \end{aligned}
    
    Note that the :math:`l_2` error is upper bounded by l. If l is too small
    this problem may be infeasible. This formulation can be converted to a
    SOCP problem.

    Args:
        m (int): Dimension of the observation vector :math:`\mathbf{b}`.
        k (int): Dimension of the sparse vector :math:`\mathbf{x}` (or the
            number of columns of the dictionary matrix, :math:`\mathbf{A}`).
        formulation (str): ``'penalizedl1'``, ``'constrainedl1'`` or
            ``'constrainedl2'``. Default value is ``'penalizedl1'``.
        nonnegative (bool): Specifies whether :math:`\mathbf{x}` must be
            nonnegative. Default value is ``False``.
    """

    def __init__(self, m, k, formulation='penalizedl1', nonnegative=False):
        if not cvx_available:
            raise RuntimeError('Cannot initialize when cvxpy is not available.')
        # Initialize parameters and variables
        A = cvx.Parameter((m, k))
        b = cvx.Parameter((m, 1))
        l = cvx.Parameter(nonneg=True)
        x = cvx.Variable((k, 1))
        # Create the problem
        if formulation == 'penalizedl1':
            obj_func = 0.5 * cvx.sum_squares(cvx.matmul(A, x) - b) + l * cvx.norm1(x)
            constraints = []
        elif formulation == 'constrainedl1':
            obj_func = cvx.sum_squares(cvx.matmul(A, x) - b)
            constraints = [cvx.norm1(x) <= l]
        elif formulation == 'constrainedl2':
            obj_func = cvx.norm1(x)
            constraints = [cvx.norm(cvx.matmul(A, x) - b) <= l]
        else:
            raise ValueError("Unknown formulation '{0}'.".format(formulation))
        if nonnegative:
            constraints.append(x >= 0)
        problem = cvx.Problem(cvx.Minimize(obj_func), constraints)
        self._formulation = formulation
        self._A = A
        self._b = b
        self._l = l
        self._x = x
        self._obj_func = obj_func
        self._constraints = constraints
        self._problem = problem

    def solve(self, A, b, l, **kwargs):
        """Solves the problem with the specified parameters.

        Args:
            A (~numpy.ndarray): Dictionary matrix.
            b (~numpy.ndarray): Observation vector.
            l (float): Regularization/constraint parameter.
            **kwargs: Other keyword arguments to be passed to the solver.
        """
        self._A.value = A
        self._b.value = b
        self._l.value = l
        self._problem.solve(**kwargs)
        if self._problem.status != 'optimal':
            warnings.warn('Optimal solution cannot be obtained.')
            return np.zeros((self._x.size,))
        return self._x.value

class L21RegularizedLeastSquaresProblem:
    r"""Creates an :math:`l_{2,1}`-norm regularized least squares problem.
    
    The :math:`l_{2,1}`-norm of a matrix variable
    :math:`\mathbf{X} \in \mathbb{C}^{K \times L}` is given by
    
    .. math::
        
        \| \mathbf{X} \|_{2,1}
        = \sum_{i=1}^K \left(\sum_{j=1}^L |X_{ij}|^2\right)^{\frac{1}{2}}.
    
    The :math:`l_{2,1}`-norm regularized least squares problem is given by

    .. math::

        \min_{\mathbf{X}}
        \frac{1}{2} \| \mathbf{A}\mathbf{X} - \mathbf{B} \|_F^2 +
        l \| \mathbf{X} \|_{2,1},
    
    where :math:`\mathbf{A}` is :math:`M \times K`, :math:`\mathbf{X}` is
    :math:`K \times L`, :math:`\mathbf{B}` is :math:`M \times L`, and :math:`l`
    is the regularization parameter. Usually :math:`\mathbf{A}` is the
    dictionary matrix, :math:`\mathbf{X}` is the sparse signal to be
    reconstructed, and :math:`\mathbf{B}` is the observation matrix where each
    column of :math:`\mathbf{B}` represents a single observation.

    Args:
        m (int): Number of rows of the dictionary matrix :math:`\mathbf{A}`.
        k (int): Number of rows of :math:`\mathbf{X}` (or the number of columns
            of the dictionary matrix, :math:`\mathbf{A}`).
        n (int): Number of observations (the number of columns of the
            observation matrix, :math:`\mathbf{B}`)
        complex (bool): Specifies whether all matrices are complex. Default
            value is ``False``.
    """

    def __init__(self, m, k, n, complex=False):
        if not cvx_available:
            raise RuntimeError('Cannot initialize when cvxpy is not available.')
        # Initialize parameters and variables
        A = cvx.Parameter((m, k), complex=complex)
        B = cvx.Parameter((m, n), complex=complex)
        l = cvx.Parameter(nonneg=True)
        X = cvx.Variable((k, n), complex=complex)
        # Create the problem
        # CVXPY issue:
        #   cvx.norm does not work if axis is not 0.
        # Workaround:
        #   use cvx.norm(X.T, 2, axis=0) instead of cvx.norm(X, 2, axis=1)
        obj_func = 0.5 * cvx.norm(cvx.matmul(A, X) - B, 'fro')**2 + \
                   l * cvx.sum(cvx.norm(X.T, 2, axis=0))
        self._problem = cvx.Problem(cvx.Minimize(obj_func))
        self._A = A
        self._B = B
        self._l = l
        self._X = X

    def solve(self, A, B, l, **kwargs):
        """Solves the problem with the specified parameters.

        Args:
            A (~numpy.ndarray): Dictionary matrix.
            B (~numpy.ndarray): Observation matrix.
            l (~numpy.ndarray): Regularization parameter.
            **kwargs: Other keyword arguments to be passed to the solver.
        """
        self._A.value = A
        self._B.value = B
        self._l.value = l
        self._problem.solve(**kwargs)
        if self._problem.status != 'optimal':
            warnings.warn('Optimal solution cannot be obtained.')
            return np.zeros(self._X.shape)
        return self._X.value

class SparseCovarianceMatching(SpectrumBasedEstimatorBase):
    r"""Creates a source location estimator based on matching the sparse
    representation of the covariance matrix.

    The sources are assumed to be uncorrelated. Take 1D far-field sources
    as an example. After discretizing the range of source locations into a fine
    grid of :math:`G` points,
    :math:`\mathbf{\theta} = [\theta_1, \ldots, \theta_G]^T`, the vectorized
    covariance matrix can be expressed as

    .. math::

        \mathbf{r} = \begin{bmatrix}
            \mathbf{A}^*(\mathbf{\theta}) \odot \mathbf{A}(\mathbf{\theta}) &
            \mathrm{vec}(\mathbf{I})
        \end{bmatrix}
        \begin{bmatrix} \mathbf{p} \\ \sigma^2 \end{bmatrix}
    
    where :math:`\mathbf{A}` is the steering matrix of the discretized source
    locations, :math:`\mathbf{p} \in \mathbb{R}_+^G` is a sparse vector of the
    source powers, and :math:`\sigma^2` is the noise variance. If all sources
    are located on the grid, :math:`p_k` should be non-zero if there is a
    source located at :math:`\theta_k` and zero otherwise.

    Let :math:`\mathbf{\Phi} = \lbrack \mathbf{A}^* \odot \mathbf{A}, \mathrm{vec}(\mathbf{I}) \rbrack`,
    and :math:`\mathbf{x} = \lbrack \mathbf{p}^T, \sigma^2 \rbrack^T`. We have
    :math:`\mathbf{r} = \mathbf{\Phi} \mathbf{x}`, where :math:`\mathbf{x}` is
    sparse. We call this expression the sparse representation of the covariance
    matrix. Given the estimate of :math:`\mathbf{r}`, :math:`\hat{\mathbf{r}}`,
    we can formulate a sparse recovery problem to recovery the sparse vector
    :math:`\mathbf{x}`, and then recover the source locations.

    Args:
        array (~doatools.model.arrays.ArrayDesign): Array design.
        wavelength (float): Wavelength of the carrier wave.
        search_grid (~doatools.estimation.grid.SearchGrid): The search grid
            used to locate the sources.
        formulation (str): ``'penalizedl1'``, ``'constrainedl1'``, or
            ``'constrainedl2'``.
        **kwargs: Other keyword arguments supported by
            :class:`~doatools.estimation.core.SpectrumBasedEstimatorBase`.

    References:
        [1] J. Yin and T. Chen, "Direction-of-Arrival Estimation Using a Sparse
        Representation of Array Covariance Vectors," IEEE Transactions on
        Signal Processing, vol. 59, no. 9, pp. 4489–4493, Sep. 2011.

        [2] Y. D. Zhang, M. G. Amin, and B. Himed, "Sparsity-based DOA
        estimation using co-prime arrays," in 2013 IEEE International
        Conference on Acoustics, Speech and Signal Processing (ICASSP),
        2013, pp. 3967-3971.
        
        [3] Z. Tan and A. Nehorai, "Sparse direction of arrival estimation using
        co-prime arrays with off-grid targets," IEEE Signal Processing
        Letters, vol. 21, no. 1, pp. 26-29, Jan. 2014.
    """

    def __init__(self, array, wavelength, search_grid, noise_known=False,
                 formulation='penalizedl1', **kwargs):
        super().__init__(array, wavelength, search_grid, **kwargs)
        self._formulation = formulation
        self._noise_known = noise_known
        # vec(R) -> m*m elements, real + image -> 2*m*m
        m = 2 * self._array.size**2
        k = self._search_grid.size
        # If noise is not known, we need a additional column for vec(I).
        if not self._noise_known:
            k += 1
        # Initialize the problem.
        self._problem = L1RegularizedLeastSquaresProblem(m, k, formulation, True)

    def _compute_atom_matrix(self, grid):
        A = self._array.steering_matrix(
            grid.source_placement, self._wavelength,
            perturbations='known'
        )
        Phi = khatri_rao(A.conj(), A)
        if not self._noise_known:
            Phi = np.hstack((Phi, vec(np.eye(self._array.size))))
        Phi = np.vstack((Phi.real, Phi.imag))
        return Phi

    def _get_sparse_spectrum(self, Phi, R, l, solver_options):
        r = vec(R)
        r = np.vstack((r.real, r.imag))
        sol = self._problem.solve(Phi, r, l, **solver_options).flatten()
        if not self._noise_known:
            # The last element is the noise variance estimate.
            # TODO: output noise estimate?
            sol = sol[:-1]
        return sol
    
    def estimate(self, R, k, l, sigma=None, solver_options={}, **kwargs):
        r"""Estimates the source locations from the given covariance matrix.

        Args:
            R (~numpy.ndarray): Covariance matrix input. The size of R must
                match that of the array design used when creating this
                estimator.
            
            k (int): Expected number of sources.
            
            l (float): The regularization parameter. The meaning of this
                parameter depends on the formulation:
                
                * ``'penalizedl1'``: regularization parameter of the l1 penalty
                  term. Larger values of ``l`` usually leads to more sparse
                  solutions, at the cost of increased biases.
                * ``'constrainedl1'``: upper bound of the l1 norm of the signal
                  vector. Smaller values of ``l`` usually leads to more sparse
                  solutions, at the cost of increased biases.
                * ``'constrainedl2'``: upper bound of the l2 norm of the
                  residual. Smaller values of ``l`` usually leads to better
                  reconstruction results. However the optimization problem
                  will become infeasible if ``l`` is too small.
                
                See :class:`L1RegularizedLeastSquaresProblem`
                for more details.
            
            solver_options (dict): A dictionary of additional keyword arguments
                to be passed to the optimizer. For instance, you can specify the
                solver or set the verbosity.
            
            return_spectrum (bool): Set to ``True`` to also output the spectrum
                for visualization. Default value if ``False``.
                
        Returns:
            A tuple with the following elements.

            * resolved (:class:`bool`): ``True`` is the solver successfully
              obtained the sparse solution and the peak finder successfully
              identified the desired number of peaks. This flag does **not**
              guarantee that the estimated source locations are correct. The
              estimated source locations may be completely wrong!
              If resolved is False, both ``estimates`` and ``spectrum`` will be
              ``None``.
            * estimates (:class:`~doatools.model.sources.SourcePlacement`):
              A :class:`~doatools.model.sources.SourcePlacement` instance of the
              same type as the one used in the search grid, represeting the
              estimated source locations. Will be ``None`` if resolved is
              ``False``.
            * spectrum (:class:`~numpy.ndarray`): An numpy array of the same
              shape of the specified search grid, consisting of values evaluated
              at the grid points. Only present if ``return_spectrum`` is
              ``True``.
        """
        if 'refine_estimates' in kwargs:
            raise ValueError('Grid refinement is not supported.')
        ensure_covariance_size(R, self._array)
        if self._noise_known:
            if sigma is None:
                raise ValueError('sigma must be specified when noise variance is assumed known.')
            # Do not modify R in-place!
            R = R - np.eye(self._array.size) * sigma
        f_sp = lambda Phi: self._get_sparse_spectrum(Phi, R, l, solver_options)
        return self._estimate(f_sp, k, **kwargs)

class GroupSparseEstimator(SpectrumBasedEstimatorBase):
    r"""Creates a group-sparsity based estimator.

    The group-sparsity based estimator considers the following mulitple
    measurement vector (MMV) model:

    .. math::
    
        \mathbf{Y} = \mathbf{A} \mathbf{X} + \mathbf{N},

    where each column of :math:`\mathbf{Y}` represents a single snapshot
    vector, each column of :math:`\mathbf{X}` represents a single source signal
    vector, each column of :math:`\mathbf{N}` represents a single noise vector.

    Similar to :class:`SparseCovarianceMatching`, we discretize the range of
    source locations into a fine grid, and :math:`\mathbf{A}` is the steering
    matrix of the discretized source locations. If we can find out the non-zero
    rows of :math:`\mathbf{X}`, we can then recover the source locations.

    The group-sparsity based estimator solves the following mixed-norm
    optimization problem:

    .. math::

        \min_{\mathbf{X}} \frac{1}{2}\| \mathbf{A}\mathbf{X} - \mathbf{Y} \|_2
        + \lambda \| \mathbf{X} \|_{2,1},

    where :math:`\lambda` is the regularization parameter, and

    .. math::
    
        \| \mathbf{X} \|_{2,1}
        =\sum_{i} \left(\sum_{j} |X_{ij}|^2\right)^{\frac{1}{2}}.
    
    After recovering :math:`\mathbf{X}`, the :math:`l_2` norms of the rows of
    :math:`\mathbf{X}` forms a pseudo spectrum. We can then find the source
    locations by identifying the largest peaks.

    Args:
        array (~doatools.model.arrays.ArrayDesign): Array design.
        wavelength (float): Wavelength of the carrier wave.
        search_grid (~doatools.estimation.grid.SearchGrid): The search grid
            used to locate the sources.
        n_snapshots (int): Number of snapshots used.
        **kwargs: Other keyword arguments supported by
            :class:`~doatools.estimation.core.SpectrumBasedEstimatorBase`.
    
    References:
        [1] D. Malioutov, M. Cetin, and A. S. Willsky, "A sparse signal
        reconstruction perspective for source localization with sensor
        arrays," IEEE Transactions on Signal Processing, vol. 53, no. 8,
        pp. 3010-3022, Aug. 2005.
    """

    def __init__(self, array, wavelength, search_grid, n_snapshots, **kwargs):
        super().__init__(array, wavelength, search_grid, **kwargs)
        self._n_snapshots = n_snapshots
        self._problem = L21RegularizedLeastSquaresProblem(
            array.size, search_grid.size, n_snapshots, True
        )
    
    def _get_sparse_spectrum(self, A, Y, l, solver_options):
        X = self._problem.solve(A, Y, l, **solver_options)
        return np.linalg.norm(X, ord=2, axis=1)

    def estimate(self, Y, k, l, solver_options={}, **kwargs):
        """Estimates the source locations from the given measurements.

        Args:
            Y (~numpy.ndarray): The matrix of measurements, each column of which
                represents a single snapshot.
            k (int): Expected number of sources.
            l (float): The regularization parameter. Larger values of ``l``
                usually leads to more sparse solutions, at the cost of increased
                biases.
            solver_options (dict): A dictionary of additional keyword arguments
                to be passed to the optimizer. For instance, you can specify the
                solver or set the verbosity.
            return_spectrum (bool): Set to ``True`` to also output the spectrum
                for visualization. Default value if ``False``.
                
        Returns:
            A tuple with the following elements.

            * resolved (:class:`bool`): ``True`` is the solver successfully
              obtained the sparse solution and the peak finder successfully
              identified the desired number of peaks. This flag does **not**
              guarantee that the estimated source locations are correct. The
              estimated source locations may be completely wrong!
              If resolved is False, both ``estimates`` and ``spectrum`` will be
              ``None``.
            * estimates (:class:`~doatools.model.sources.SourcePlacement`):
              A :class:`~doatools.model.sources.SourcePlacement` instance of the
              same type as the one used in the search grid, represeting the
              estimated source locations. Will be ``None`` if resolved is
              ``False``.
            * spectrum (:class:`~numpy.ndarray`): An numpy array of the same
              shape of the specified search grid, consisting of values evaluated
              at the grid points. Only present if ``return_spectrum`` is
              ``True``.
        """
        if 'refine_estimates' in kwargs:
            raise ValueError('Grid refinement is not supported.')
        if Y.shape[0] != self._array.size:
            raise ValueError('The number of rows of Y must be equal to the array size.')
        if Y.shape[1] != self._n_snapshots:
            raise ValueError('The number of columns of Y must be equal to the number of snapshots.')
        f_sp = lambda A: self._get_sparse_spectrum(A, Y, l, solver_options)
        return self._estimate(f_sp, k, **kwargs)
