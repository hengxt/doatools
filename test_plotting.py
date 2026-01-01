import numpy as np
import doatools.model as model
import doatools.estimation as estimation
import doatools.performance as perf
import doatools.plotting as plt_tools

# 测试脚本，演示新的绘图函数的使用

print("=" * 60)
print("DOATools Plotting Functions Test")
print("=" * 60)

# 参数设置
wavelength = 1.0
d0 = wavelength / 2
array = model.UniformLinearArray(8, d0)

true_angles = np.array([-np.pi/6, np.pi/6])
sources = model.FarField1DSourcePlacement(true_angles)

# 创建一些简单的估计器
estimators = {
    'RootMUSIC': estimation.RootMUSIC1D(wavelength),
    'MUSIC': estimation.MUSIC(array, wavelength, estimation.FarField1DSearchGrid(
        start=-np.pi/2, stop=np.pi/2, size=500, unit='rad'
    ))
}

# 运行一个简单的性能评估来生成测试数据
print("\n1. Generating test data...")
snr = 0.0
n_snapshots = 100
n_monte_carlo = 50

result = perf.evaluate_performance(
    array=array,
    sources=sources,
    snr=snr,
    n_snapshots=n_snapshots,
    n_monte_carlo=n_monte_carlo,
    estimators=estimators,
    crb_types=['sto'],
    metrics=['mse', 'bias']
)

print(f"   ✓ Generated performance results for {len(estimators)} algorithms")
print(f"   ✓ CRB values: {result.crb_results}")

# 生成一些模拟的估计角度数据，用于测试散点图、CDF等
print("\n2. Generating simulated estimates for visualization...")

# 模拟估计角度，添加一些噪声
np.random.seed(42)
estimates = {}
for alg_name in estimators.keys():
    # 为每个算法生成n_monte_carlo次估计，添加不同程度的噪声
    if alg_name == 'RootMUSIC':
        noise = np.random.normal(0, 0.02, (n_monte_carlo, len(true_angles)))
    else:
        noise = np.random.normal(0, 0.03, (n_monte_carlo, len(true_angles)))
    estimates[alg_name] = np.tile(true_angles, (n_monte_carlo, 1)) + noise

print(f"   ✓ Generated {n_monte_carlo} Monte Carlo estimates for each algorithm")

# 生成SNR扫描测试数据
print("\n3. Generating SNR sweep data...")
snrs = np.linspace(-10, 10, 5)  # 5个SNR值，从-10dB到10dB

# 为每个SNR生成模拟的MSE值
mse_results = {
    'RootMUSIC': [],
    'MUSIC': [],
    'ESPRIT': [],
    'MVDR': []
}

# 生成模拟的CRB值
crb_values = 10**(-0.1*snrs - 3)  # 模拟CRB随SNR的变化

for snr in snrs:
    # 为每个算法生成模拟的MSE值
    mse_results['RootMUSIC'].append(10**(-0.1*snr - 2.5) + np.random.normal(0, 0.1e-3))
    mse_results['MUSIC'].append(10**(-0.1*snr - 2.3) + np.random.normal(0, 0.15e-3))
    mse_results['ESPRIT'].append(10**(-0.1*snr - 2.4) + np.random.normal(0, 0.12e-3))
    mse_results['MVDR'].append(10**(-0.1*snr - 1.8) + np.random.normal(0, 0.2e-3))

# 将列表转换为numpy数组
for alg in mse_results:
    mse_results[alg] = np.array(mse_results[alg])

print(f"   ✓ Generated MSE results for {len(mse_results)} algorithms across {len(snrs)} SNR values")

print("\n" + "=" * 60)
print("Testing Plotting Functions")
print("=" * 60)

# 测试1: plot_metric_vs_parameter (折线图)
print("\n1. Testing plot_metric_vs_parameter function...")
plt_tools.plot_metric_vs_parameter(
    parameter_values=snrs,
    results=mse_results,
    parameter_name='SNR',
    metric_name='MSE',
    parameter_unit='dB',
    metric_unit='rad²',
    title='MSE vs. SNR for Multiple DOA Algorithms',
    show_crb=True,
    crb_values=crb_values,
    crb_label='Stochastic CRB'
)
print("   ✓ plot_metric_vs_parameter function test completed")

# 测试2: plot_scatter_estimates (散点图)
print("\n2. Testing plot_scatter_estimates function...")
plt_tools.plot_scatter_estimates(
    true_angles=true_angles,
    estimates=estimates['RootMUSIC'],
    algorithm_name='RootMUSIC',
    angle_unit='rad'
)
print("   ✓ plot_scatter_estimates function test completed")

# 测试3: plot_cdf (累积分布函数图)
print("\n3. Testing plot_cdf function...")
plt_tools.plot_cdf(
    estimates=estimates['MUSIC'],
    true_angles=true_angles,
    algorithm_name='MUSIC',
    metric_name='Absolute Error',
    metric_unit='rad'
)
print("   ✓ plot_cdf function test completed")

# 测试4: plot_histogram (直方图)
print("\n4. Testing plot_histogram function...")
plt_tools.plot_histogram(
    estimates=estimates['RootMUSIC'],
    true_angles=true_angles,
    algorithm_name='RootMUSIC',
    bins=30,
    metric_unit='rad'
)
print("   ✓ plot_histogram function test completed")

# 测试5: plot_resolution_comparison (分辨率比较图)
print("\n5. Testing plot_resolution_comparison function...")

# 生成模拟的分辨率数据
delta_thetas = np.linspace(0.5, 5.0, 10)
# 转换为弧度
delta_thetas_rad = np.deg2rad(delta_thetas)

# 为不同算法生成模拟的成功分辨率
success_rates = {
    'RootMUSIC': 1.0 / (1.0 + np.exp(-2.0*(delta_thetas - 2.0))),
    'MUSIC': 1.0 / (1.0 + np.exp(-1.5*(delta_thetas - 2.5))),
    'ESPRIT': 1.0 / (1.0 + np.exp(-1.8*(delta_thetas - 2.2))),
    'MVDR': 1.0 / (1.0 + np.exp(-1.2*(delta_thetas - 3.0))),
    'Bartlett': 1.0 / (1.0 + np.exp(-0.8*(delta_thetas - 4.0)))
}

plt_tools.plot_resolution_comparison(
    delta_thetas=delta_thetas,
    success_rates=success_rates,
    parameter_unit='deg',
    title='Resolution Comparison for Multiple DOA Algorithms'
)
print("   ✓ plot_resolution_comparison function test completed")

print("\n" + "=" * 60)
print("All plotting functions tests completed successfully!")
print("=" * 60)
