import unittest
from doatools.model.arrays import UniformLinearArray
from doatools.model.sources import FarField1DSourcePlacement
from doatools.model.signals import ComplexStochasticSignal
from doatools.model import get_narrowband_snapshots
import numpy as np
from doatools.estimation import Interferometer1D



class TestInterferometer(unittest.TestCase):

    def setUp(self):
        self.wavelength = 1.

    def test_music_1d(self):
        np.random.seed(42)
        ula = UniformLinearArray(10, self.wavelength / 2)
        n_sources = 2
        sources = FarField1DSourcePlacement(np.linspace(-np.pi/8, np.pi/3, n_sources))
        power_source = 1.0
        snr = 0
        # power_noise = power_source / (10 ** (snr / 10))
        power_noise = 0
        source_signal = ComplexStochasticSignal(sources.size, power_source)
        noise_signal = ComplexStochasticSignal(ula.size, power_noise)
        snapshots = 500

        Y, R = get_narrowband_snapshots(ula, sources, self.wavelength, source_signal, noise_signal, snapshots, True)
        interferometer = Interferometer1D(self.wavelength)
        print(ula.element_locations[:, 0])
        resolved, estimates = interferometer.estimate(Y, ula.element_locations[:, 0])

        print(resolved)
        print(estimates.locations)
        print(sources.locations)
        # self.assertTrue(resolved)
        # npt.assert_allclose(estimates.locations, sources.locations, rtol=1e-6, atol=1e-8)


if __name__ == '__main__':
    unittest.main()