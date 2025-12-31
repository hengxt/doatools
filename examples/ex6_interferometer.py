from doatools.estimation.music import RootMUSIC1D
from doatools.model.arrays import UniformLinearArray
from doatools.model.sources import FarField1DSourcePlacement
from doatools.model.signals import ComplexStochasticSignal
from doatools.model import get_narrowband_snapshots
import numpy as np
from doatools.estimation import Interferometer1D
import matplotlib.pyplot as plt



def ComparewithMusic():
    # 参数设置
    wavelength = 1.0
    n_elements = 10
    d = wavelength / 2
    ula = UniformLinearArray(n_elements, d)
    n_snapshots = 100
    snr_db = 10
    n_trials = 100
    angle_range_deg = np.arange(-60, 60.1, 0.5)
    angle_range_rad = np.deg2rad(angle_range_deg)

    rmse_rootmusic = []
    rmse_interf = []

    for theta_true in angle_range_rad:
        errors_rootmusic = []
        errors_interf = []
        for _ in range(n_trials):
            # 单信号源
            sources = FarField1DSourcePlacement([theta_true])
            power_source = 1.0
            power_noise = power_source / (10 ** (snr_db / 10))
            source_signal = ComplexStochasticSignal(sources.size, power_source)
            noise_signal = ComplexStochasticSignal(ula.size, power_noise)
            Y, R = get_narrowband_snapshots(ula, sources, wavelength, source_signal, noise_signal, n_snapshots, True)

            # RootMUSIC
            rmusic = RootMUSIC1D(wavelength)
            resolved_r, estimates_r = rmusic.estimate(R, 1, ula.d0, unit='rad')
            if resolved_r and estimates_r is not None:
                est_angle_r = estimates_r.locations[0]
                # 处理周期性误差
                err_r = np.arctan2(np.sin(est_angle_r - theta_true), np.cos(est_angle_r - theta_true))
                errors_rootmusic.append(err_r)
            # Interferometer
            interf = Interferometer1D(wavelength)
            resolved_i, estimates_i = interf.estimate(Y, ula.element_locations, unit='rad')
            if resolved_i and estimates_i is not None:
                est_angle_i = estimates_i.locations
                err_i = np.arctan2(np.sin(est_angle_i - theta_true), np.cos(est_angle_i - theta_true))
                errors_interf.append(err_i)
        # 计算RMSE
        rmse_rootmusic.append(np.sqrt(np.mean(np.square(errors_rootmusic))) if errors_rootmusic else np.nan)
        rmse_interf.append(np.sqrt(np.mean(np.square(errors_interf))) if errors_interf else np.nan)

    plt.figure(figsize=(8, 5))
    plt.plot(angle_range_deg, rmse_rootmusic, label='RootMUSIC', lw=2)
    plt.plot(angle_range_deg, rmse_interf, label='Interferometer1D', lw=2)
    plt.xlabel('True DOA (deg)')
    plt.ylabel('RMSE (radian)')
    plt.title('RMSE of DOA Estimation vs True Angle (SNR=10dB, 100 MC)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def CompareSNRwithMusic():
    # 参数设置
    wavelength = 1.0
    n_elements = 10
    d = wavelength / 2
    ula = UniformLinearArray(n_elements, d)
    n_snapshots = 100
    n_trials = 100
    theta_true_deg = 32.1
    theta_true_rad = np.deg2rad(theta_true_deg)
    snr_db_range = np.arange(-10, 30.1, 2)

    rmse_rootmusic = []
    rmse_interf = []

    for snr_db in snr_db_range:
        errors_rootmusic = []
        errors_interf = []
        for _ in range(n_trials):
            sources = FarField1DSourcePlacement([theta_true_rad])
            power_source = 1.0
            power_noise = power_source / (10 ** (snr_db / 10))
            source_signal = ComplexStochasticSignal(sources.size, power_source)
            noise_signal = ComplexStochasticSignal(ula.size, power_noise)
            Y, R = get_narrowband_snapshots(ula, sources, wavelength, source_signal, noise_signal, n_snapshots, True)

            # RootMUSIC
            rmusic = RootMUSIC1D(wavelength)
            resolved_r, estimates_r = rmusic.estimate(R, 1, ula.d0, unit='rad')
            if resolved_r and estimates_r is not None:
                est_angle_r = estimates_r.locations[0]
                err_r = np.arctan2(np.sin(est_angle_r - theta_true_rad), np.cos(est_angle_r - theta_true_rad))
                errors_rootmusic.append(err_r)
            # Interferometer
            interf = Interferometer1D(wavelength)
            resolved_i, estimates_i = interf.estimate(Y, ula.element_locations, unit='rad')
            if resolved_i and estimates_i is not None:
                est_angle_i = estimates_i.locations
                err_i = np.arctan2(np.sin(est_angle_i - theta_true_rad), np.cos(est_angle_i - theta_true_rad))
                errors_interf.append(err_i)
        rmse_rootmusic.append(np.sqrt(np.mean(np.square(errors_rootmusic))) if errors_rootmusic else np.nan)
        rmse_interf.append(np.sqrt(np.mean(np.square(errors_interf))) if errors_interf else np.nan)

    plt.figure(figsize=(8, 5))
    plt.plot(snr_db_range, rmse_rootmusic, label='RootMUSIC', lw=2)
    plt.plot(snr_db_range, rmse_interf, label='Interferometer1D', lw=2)
    plt.xlabel('SNR (dB)')
    plt.ylabel('RMSE (radian)')
    plt.yscale('log')
    plt.title(f'RMSE vs SNR (True DOA={theta_true_deg}°, 100 MC)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


CompareSNRwithMusic()