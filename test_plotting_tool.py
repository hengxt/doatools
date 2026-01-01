import numpy as np
import matplotlib.pyplot as plt
from doatools.plotting.plot_performance import (
    get_doa_method_style, set_doa_method_style, get_global_config,
    plot_metric_vs_parameter, plot_resolution_comparison
)

# 测试1: 验证默认配置和样式获取
print("测试1: 验证默认配置和样式获取")
print("全局配置:", get_global_config())

# 测试已知方法的样式
methods = ['RootMUSIC', 'MUSIC', 'ESPRIT', 'MVDR', 'Bartlett', 'MinNorm']
for method in methods:
    color, marker = get_doa_method_style(method)
    print(f"{method}: 颜色={color}, 标记={marker}")

# 测试未知方法的样式
color, marker = get_doa_method_style('UnknownMethod', 2)
print(f"UnknownMethod: 颜色={color}, 标记={marker}")

# 测试2: 验证样式修改
print("\n测试2: 验证样式修改")
set_doa_method_style('RootMUSIC', color='red', marker='s')
color, marker = get_doa_method_style('RootMUSIC')
print(f"修改后 RootMUSIC: 颜色={color}, 标记={marker}")

# 测试3: 验证绘图功能
print("\n测试3: 验证绘图功能")

# 生成测试数据
parameter_values = np.linspace(-20, 30, 11)
results = {
    'RootMUSIC': 10**(-parameter_values/10),
    'MUSIC': 1.5 * 10**(-parameter_values/10),
    'ESPRIT': 2 * 10**(-parameter_values/10),
    'MVDR': 5 * 10**(-parameter_values/10),
    'Bartlett': 10 * 10**(-parameter_values/10)
}

# 绘制测试图
fig, ax = plt.subplots(figsize=(10, 6))
plot_metric_vs_parameter(parameter_values, results, 'SNR', 'MSE', 
                         parameter_unit='dB', metric_unit='rad²',
                         title='Test Plot: MSE vs. SNR', ax=ax)
plt.savefig('test_plot_metric.png')
print("保存测试图: test_plot_metric.png")

# 测试分辨率比较图
print("\n测试4: 验证分辨率比较图")
delta_thetas = np.linspace(0.01, 0.1, 10)
success_rates = {
    'RootMUSIC': 1 / (1 + np.exp(-50*(delta_thetas - 0.05))),
    'MUSIC': 1 / (1 + np.exp(-40*(delta_thetas - 0.06))),
    'ESPRIT': 1 / (1 + np.exp(-30*(delta_thetas - 0.07)))
}

fig, ax = plt.subplots(figsize=(10, 6))
plot_resolution_comparison(delta_thetas, success_rates, parameter_unit='rad',
                           title='Test Plot: Resolution Comparison', ax=ax)
plt.savefig('test_plot_resolution.png')
print("保存测试图: test_plot_resolution.png")

print("\n所有测试完成!")
