# Performance analysis of a uniform linear array using the enhanced evaluator
# This script demonstrates how to use the improved performance evaluation tool
# with multiple estimators, multiple metrics, and multiple CRB types

import numpy as np
import doatools.model as model
import doatools.estimation as estimation
import doatools.performance as perf

# Parameters setup
wavelength = 1.0  # normalized

# Create a 12-element ULA
array = model.UniformLinearArray(12, wavelength / 2)

# Place 8 sources uniformly within (-pi/3, pi/4)
sources = model.FarField1DSourcePlacement(
    np.linspace(-np.pi/3, np.pi/4, 8)
)

# Set up multiple estimators
estimators = {
    'RootMUSIC': estimation.RootMUSIC1D(wavelength),
    'MUSIC': estimation.MUSIC(array, wavelength, estimation.FarField1DSearchGrid(
        start=-np.pi/2, stop=np.pi/2, size=500, unit='rad'
    ))
}

# Test parameters
SNR = 0.0  # dB
n_snapshots = 200
n_monte_carlo = 50  # Reduced for faster demonstration

print("=== DOA Performance Evaluation with Multiple Settings ===")
print(f"Array: {array.__class__.__name__} with {array.size} elements")
print(f"Sources: {sources.size} far-field sources")
print(f"Estimators: {', '.join(estimators.keys())}")
print(f"SNR: {SNR} dB")
print(f"Snapshots: {n_snapshots}")
print(f"Monte Carlo runs: {n_monte_carlo}")
print()

# Example 1: Multiple estimators, multiple CRB types, multiple metrics
print("1. Using multiple estimators, CRB types, and metrics:")
print("   Estimators: RootMUSIC, MUSIC")
print("   CRB types: sto, det, stouc")
print("   Metrics: mse, rmse")
print()

result = perf.evaluate_performance(
    array=array,
    sources=sources,
    snr=SNR,
    n_snapshots=n_snapshots,
    n_monte_carlo=n_monte_carlo,
    estimators=estimators,
    crb_types=['sto', 'det', 'stouc'],
    metrics=['mse', 'rmse']
)

# Print the result
print(result)
print()

# Example 2: Using list of estimators (instead of dictionary)
print("2. Using list of estimators:")
print("   Estimators: RootMUSIC, MUSIC")
print("   CRB types: sto")
print("   Metrics: mse")
print()

# Create estimators as list
estimator_list = [
    estimation.RootMUSIC1D(wavelength),
    estimation.MUSIC(array, wavelength, estimation.FarField1DSearchGrid(
        start=-np.pi/2, stop=np.pi/2, size=500, unit='rad'
    ))
]

result2 = perf.evaluate_performance(
    array=array,
    sources=sources,
    snr=SNR,
    n_snapshots=n_snapshots,
    n_monte_carlo=n_monte_carlo,
    estimators=estimator_list,
    crb_types='sto',
    metrics='mse'
)

# Print the result
print(result2)
print()

# Example 3: Using single estimator, multiple metrics
print("3. Using single estimator with multiple metrics:")
print("   Estimator: RootMUSIC")
print("   CRB types: sto, stouc")
print("   Metrics: mse, rmse")
print()

result3 = perf.evaluate_performance(
    array=array,
    sources=sources,
    snr=SNR,
    n_snapshots=n_snapshots,
    n_monte_carlo=n_monte_carlo,
    estimators=estimation.RootMUSIC1D(wavelength),
    crb_types=['sto', 'stouc'],
    metrics=['mse', 'rmse']
)

# Print the result
print(result3)
print()

print("=== Evaluation Complete ===")
