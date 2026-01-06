# DoaTools (Revived & Enhanced)

> **An actively maintained fork of the original project [morriswmz/doatools.py](https://github.com/morriswmz/doatools.py).**
>
> The original project is an open-source project under the MIT license, but has not been updated since approximately 2018. This project aims to modernize, enhance, and maintain it long-term.

---

**📄 Read this in Chinese: [中文版本](README_CN.md)**

---

## 📖 Project Origin

This project is forked from [morriswmz/doatools.py](https://github.com/morriswmz/doatools.py), a Python library for **Direction of Arrival (DoA) estimation**.

Given that the original project has been unmaintained for many years, this project aims to:
1.  **Refactor and optimize code**: Improve readability, maintainability, and computational efficiency.
2.  **Extend algorithms and functionality**: Incorporate more modern DoA estimation algorithms and utilities.
3.  **Improve documentation and examples**: Provide clearer usage instructions and practical application examples.

## ✨ Core Features

### Array Design
- Supports various array models: ULA (Uniform Linear Array), UCA (Uniform Circular Array), CPA (CoPrime Array), Nested Array, etc.
- Visualization of array geometry and difference coarray
- Support for location error modeling and analysis

### DoA Estimation Algorithms
- **Subspace methods**: MUSIC, Root-MUSIC, ESPRIT, MinNorm
- **Beamforming methods**: MVDR, Bartlett
- **Sparse methods**: Sparse covariance matching, Group sparse estimation
- **Maximum likelihood methods**: Asymptotic ML, Conditional ML, Weighted subspace fitting
- **Interferometer method**: 1D interferometer

### Covariance Matrix Reconstruction (Sparse Array Enhancement)
- **SPA**: Spatial-smoothing based Augmentation via semidefinite programming
- **ANM**: Covariance matrix reconstruction via nuclear norm minimization
- **StructCovMLE**: Structured covariance maximum likelihood estimation
- **Wasserstein**: Covariance matrix reconstruction based on Wasserstein distance

### Performance Evaluation
- Monte Carlo simulation evaluation framework
- Multi-metric evaluation: MSE, RMSE, MAE, Bias
- Support for multiple CRB calculations: Stochastic CRB, Deterministic CRB, Stochastic Uncorrelated CRB
- Support for parallel computing acceleration
- Performance visualization tools: Parameter-performance curves, scatter plots, CDF curves

## 📁 Code Structure

```
doatools/
├── model/               # Array models and signal models
├── estimation/          # DoA estimation algorithms
│   ├── coarray_reconstruction.py   # Covariance matrix reconstruction algorithms
│   ├── music.py        # MUSIC-related algorithms
│   ├── beamforming.py  # Beamforming methods
│   └── ...
├── performance/         # Performance evaluation module
│   ├── evaluator.py     # Performance evaluator
│   └── crb.py          # Cramér-Rao Bound
├── plotting/           # Visualization tools
│   ├── plot_performance.py  # Performance plotting functions
│   └── ...
└── utils/              # Utility functions
```

## ⚡ Completed Updates

- **[Completed]** Implemented covariance matrix reconstruction algorithms (SPA, ANM, StructCovMLE, Wasserstein)
- **[Completed]** Implemented performance evaluation framework (Monte Carlo simulation, multi-metric evaluation)
- **[Completed]** Implemented performance visualization tools (curves, scatter plots, CDF plots)
- **[Completed]** Improved example notebooks covering array design, DoA estimation, and performance evaluation

## 🚀 Quick Start

### Installation

```bash
# Install the development version from this repository
pip install git+https://github.com/hengxt/doatools.git
```

### Basic Usage

```python
import numpy as np
import doatools.model as model
import doatools.estimation as estimation

# Create array
wavelength = 1.0
d0 = wavelength / 2
ula = model.UniformLinearArray(12, d0)

# Create sources
sources = model.FarField1DSourcePlacement(np.linspace(-np.pi/4, np.pi/4, 3))

# Generate observation data
source_signal = model.ComplexStochasticSignal(sources.size, 1.0)
noise_signal = model.ComplexStochasticSignal(ula.size, 0.1)
_, R = model.get_narrowband_snapshots(ula, sources, wavelength, 
                                       source_signal, noise_signal, 100,
                                       return_covariance=True)

# Estimate DOA using MUSIC algorithm
grid = estimation.FarField1DSearchGrid()
music = estimation.MUSIC(ula, wavelength, grid)
resolved, estimates, spectrum = music.estimate(R, sources.size, return_spectrum=True)
print(f"MUSIC Estimates: {np.rad2deg(estimates.locations)}")
print(f"Ground truth: {np.rad2deg(sources.locations)}")
```

### Performance Evaluation Example

```python
from doatools.performance import evaluate_performance

# Define estimator
root_music = estimation.RootMUSIC1D(wavelength)

# Run performance evaluation
result = evaluate_performance(
    array=ula,
    sources=sources,
    snr=10,           # Signal-to-noise ratio 10 dB
    n_snapshots=100,  # Number of snapshots
    n_monte_carlo=100, # Number of Monte Carlo runs
    estimators=root_music,
    crb_types=['sto'],
    metrics=['mse', 'rmse'],
    verbose=1
)

print(result)
```

## 📚 Example Tutorials

This project provides a series of Jupyter Notebook tutorials:

| File | Description |
|------|-------------|
| [ex0_design_arrays.ipynb](examples/ex0_design_arrays.ipynb) | Array design and visualization |
| [ex1_doa_with_ula.ipynb](examples/ex1_doa_with_ula.ipynb) | DoA estimation using ULA |
| [ex2_sparse_array.ipynb](examples/ex2_sparse_array.ipynb) | Sparse array and covariance reconstruction |
| [ex3_performance.ipynb](examples/ex3_performance.ipynb) | DoA algorithm performance evaluation |
| [ex4_coherent_signal.ipynb](examples/ex4_coherent_signal.ipynb) | Coherent signal source processing |
| [ex5_covariance_reconstruction.ipynb](examples/ex5_covariance_reconstruction.ipynb) | Covariance matrix reconstruction algorithms |
| [ex6_performance_sdp.ipynb](examples/ex6_performance_sdp.ipynb) | SDP-based performance analysis |

Run examples:

```bash
cd examples
jupyter notebook
```

## 📄 License

Like the original project, this project follows the **MIT License**. See the [LICENSE](LICENSE) file for details.

The copyright of the original project belongs to its original author [morriswmz](https://github.com/morriswmz). The copyright of modifications made in this project belongs to the contributors of this project.

## 🙏 Acknowledgments

Sincere thanks to the original project author [morriswmz](https://github.com/morriswmz) for the foundational work, providing a valuable starting point for the community.

## 📖 References

### Covariance Matrix Reconstruction Algorithms

1. Z. Yang, L. Xie, and C. Zhang, "A Discretization-Free Sparse and Parametric Approach for Linear Array Signal Processing," *IEEE Transactions on Signal Processing*, vol. 62, no. 19, pp. 4959-4973, Oct. 2014.

2. C. Zhou, Y. Gu, X. Fan, Z. Shi, G. Mao, and Y. D. Zhang, "Direction-of-Arrival Estimation for Coprime Array via Virtual Array Interpolation," *IEEE Transactions on Signal Processing*, vol. 66, no. 22, pp. 5956-5971, Nov. 2018.

3. X. Wu, W.-P. Zhu, and J. Yan, "A Toeplitz Covariance Matrix Reconstruction Approach for Direction-of-Arrival Estimation," *IEEE Transactions on Vehicular Technology*, vol. 66, no. 9, pp. 8223-8237, Sept. 2017.

4. M. Wang, Z. Zhang, and A. Nehorai, "Grid-Less DOA Estimation Using Sparse Linear Arrays Based on Wasserstein Distance," *IEEE Signal Processing Letters*, vol. 26, no. 6, pp. 838-842, June 2019.

5. R. R. Pote and B. D. Rao, "Maximum Likelihood-Based Gridless DoA Estimation Using Structured Covariance Matrix Recovery and SBL With Grid Refinement," *IEEE Transactions on Signal Processing*, vol. 71, pp. 802-815, 2023.
