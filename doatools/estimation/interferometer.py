import numpy as np
from ..model.sources import FarField1DSourcePlacement

class Interferometer1D:
    """Creates an interferometer-based DOA estimator for 1D linear arrays.

    This class implements a DOA estimation algorithm based on phase differences
    between array elements, with optional weighted least-squares estimation.
    Supports both uniform and non-uniform linear arrays.
    """

    def __init__(self, wavelength):
        """Initialize the interferometer DOA estimator.

        Args:
            wavelength (float): Wavelength of the signal in meters.
        """
        self._wavelength = wavelength

    def estimate(self, signal_matrix, element_positions, weighted='default', unit='rad'):
        """Estimate the direction-of-arrival (DOA) using the interferometer algorithm.

        Args:
            signal_matrix (~numpy.ndarray): Received signal matrix of shape (M, N), where M is
                the number of array elements and N is the number of snapshots.
            element_positions (~numpy.ndarray): Array element positions (M,) in meters.
            weighted (str or bool): Specifies whether to use weighted least-squares.
                - 'default' or True: Use weighted least-squares with weights as the
                  square of inter-element spacings (Cramer-Rao bound optimal).
                - False: Use unweighted least-squares.
                Default value is 'default'.
            unit (str): Unit of the estimated DOA. Either 'rad' or 'deg'.
                Default value is 'rad'.

        Returns:
            A tuple with the following elements:
            * resolved (:class:`bool`): True if the DOA estimation was successful.
            * estimates (:class:`~doatools.model.sources.FarField1DSourcePlacement`):
              A FarField1DSourcePlacement object with the estimated DOA.
              None if resolved is False.
        """
        # 兼容 element_locations 传入二维数组的情况
        if isinstance(element_positions, np.ndarray) and element_positions.ndim == 2:
            element_positions = element_positions[:, 0]
        # Parameter validation
        M, N = signal_matrix.shape
        
        if len(element_positions) != M:
            raise ValueError("Length of position vector element_positions must match number of rows in signal_matrix.")
        if M < 2:
            raise ValueError("At least two array elements are required.")
        if weighted not in [True, False, 'default']:
            raise ValueError("weighted must be 'default', True, or False.")
        if unit not in ['rad', 'deg']:
            raise ValueError("unit must be 'rad' or 'deg'.")

        # Compute inter-element spacings
        d = np.diff(element_positions)

        # 1. Compute phase differences between adjacent elements
        phase_diff = np.zeros(M - 1)
        for i in range(M - 1):
            R = np.mean(signal_matrix[i, :] * np.conj(signal_matrix[i + 1, :]))
            phase_diff[i] = np.angle(R)

        # 2. Hierarchical phase unwrapping (if more than two elements)
        unwrapped_phase = phase_diff.copy()
        if M > 2:
            sort_idx = np.argsort(d)
            sorted_d = d[sort_idx]
            sorted_phase = phase_diff[sort_idx]
            base_sin_theta = self._wavelength * sorted_phase[0] / (2 * np.pi * sorted_d[0])
            base_sin_theta = np.clip(base_sin_theta, -1, 1)
            for k in range(1, len(sorted_d)):
                pred_phase = 2 * np.pi * sorted_d[k] * base_sin_theta / self._wavelength
                phase_error = sorted_phase[k] - pred_phase
                n = np.round(phase_error / (2 * np.pi))
                unwrapped_phase[sort_idx[k]] = sorted_phase[k] - 2 * np.pi * n
                current_sin_theta = self._wavelength * unwrapped_phase[sort_idx[k]] / (2 * np.pi * sorted_d[k])
                base_sin_theta = (base_sin_theta + current_sin_theta) / 2
                base_sin_theta = np.clip(base_sin_theta, -1, 1)

        # 3. Least-squares estimation (严格按照正规方程)
        A_matrix = d.reshape(-1, 1)  # (M-1, 1)
        b_vector = unwrapped_phase * self._wavelength / (2 * np.pi)
        b_vector = b_vector.reshape(-1, 1)
        if weighted == 'default' or weighted is True:
            weights = np.diag(d ** 2)
            # (A' W A)^{-1} (A' W b)
            At_W = A_matrix.T @ weights
            sin_theta = np.linalg.solve(At_W @ A_matrix, At_W @ b_vector)
        else:
            AtA = A_matrix.T @ A_matrix
            Atb = A_matrix.T @ b_vector
            sin_theta = np.linalg.solve(AtA, Atb)
        # 方向修正
        sin_theta = -sin_theta
        # clip 并取标量，消除 DeprecationWarning
        sin_theta = np.clip(sin_theta, -1, 1)
        sin_theta = float(sin_theta.item())
        theta = np.arcsin(sin_theta)
        if unit == 'deg':
            theta = np.degrees(theta)
        return True, FarField1DSourcePlacement(theta, unit=unit)
