# DoaTools (Revived & Enhanced)

> **这是一个基于原项目 [morriswmz/doatools.py](https://github.com/morriswmz/doatools.py) 的活跃维护分支。**
>
> 原项目为使用MIT协议的开源项目，但由于已长期未更新（最后维护于约2018年），本项目旨在对其进行现代化重构、功能增强和长期维护。

---

**📄 Read this in English: [English Version](README_EN.md)**

---

## 📖 项目起源

本项目派生（Fork）自 [morriswmz/doatools.py](https://github.com/morriswmz/doatools.py)，一个用于**波达方向（Direction of Arrival, DoA）估计**的Python工具库。

鉴于原项目已停止维护多年，本项目计划：
1.  **代码重构与优化**：提升代码可读性、可维护性和计算效率。
2.  **扩展算法与功能**：引入更多现代DoA估计算法及实用功能。
3.  **完善文档与示例**：提供更清晰的使用说明和实际应用案例。

## ✨ 核心功能

### 阵列设计
- 支持多种阵列模型：ULA（均匀线阵）、UCA（均匀圆阵）、CPA（共阵）、Nested阵等
- 可视化阵列几何结构及差分协阵列（Difference Coarray）
- 支持位置误差建模与分析

### DoA估计算法
- **子空间方法**：MUSIC、Root-MUSIC、ESPRIT、MinNorm
- **波束形成方法**：MVDR、Bartlett
- **稀疏方法**：稀疏协方差匹配、组稀疏估计
- **最大似然方法**：渐近最大似然、条件最大似然、加权子空间拟合
- **干涉仪方法**：一维干涉仪

### 协方差矩阵重建（稀疏阵列增强）
- **SPA**：基于半定规划的空间平滑增强算法
- **ANM**：基于核范数最小化的协方差矩阵重建
- **StructCovMLE**：基于结构化协方差的最大似然估计
- **Wasserstein**：基于Wasserstein距离的协方差矩阵重建

### 性能评估
- Monte Carlo仿真评估框架
- 多指标评估：MSE、RMSE、MAE、Bias
- 支持多种CRB计算：随机CRB、确定CRB、随机不相关CRB
- 支持并行计算加速
- 性能可视化工具：参数-性能曲线、散点图、CDF曲线

## 📁 代码结构

```
doatools/
├── model/               # 阵列模型与信号模型
├── estimation/          # DoA估计算法
│   ├── coarray_reconstruction.py   # 协方差矩阵重建算法
│   ├── music.py        # MUSIC相关算法
│   ├── beamforming.py  # 波束形成方法
│   └── ...
├── performance/         # 性能评估模块
│   ├── evaluator.py     # 性能评估器
│   └── crb.py          # 克拉美-罗下界
├── plotting/           # 可视化工具
│   ├── plot_performance.py  # 性能绘图函数
│   └── ...
└── utils/              # 工具函数
```

## ⚡ 已完成更新

- **[已完成]** 实现协方差矩阵重建算法（SPA、ANM、StructCovMLE、Wasserstein）
- **[已完成]** 实现性能评估框架（Monte Carlo仿真、多指标评估）
- **[已完成]** 实现性能可视化工具（曲线图、散点图、CDF图）
- **[已完成]** 完善示例notebook，涵盖阵列设计、DoA估计、性能评估全流程

## 🚀 快速开始

### 安装

```bash
# 从本仓库安装开发版本
pip install git+https://github.com/hengxt/doatools.git
```

### 基本用法

```python
import numpy as np
import doatools.model as model
import doatools.estimation as estimation

# 创建阵列
wavelength = 1.0
d0 = wavelength / 2
ula = model.UniformLinearArray(12, d0)

# 创建信号源
sources = model.FarField1DSourcePlacement(np.linspace(-np.pi/4, np.pi/4, 3))

# 生成观测数据
source_signal = model.ComplexStochasticSignal(sources.size, 1.0)
noise_signal = model.ComplexStochasticSignal(ula.size, 0.1)
_, R = model.get_narrowband_snapshots(ula, sources, wavelength, 
                                       source_signal, noise_signal, 100,
                                       return_covariance=True)

# 使用MUSIC算法估计DOA
grid = estimation.FarField1DSearchGrid()
music = estimation.MUSIC(ula, wavelength, grid)
resolved, estimates, spectrum = music.estimate(R, sources.size, return_spectrum=True)
print(f"MUSIC Estimates: {np.rad2deg(estimates.locations)}")
print(f"Ground truth: {np.rad2deg(sources.locations)}")
```

### 性能评估示例

```python
from doatools.performance import evaluate_performance

# 定义估计器
root_music = estimation.RootMUSIC1D(wavelength)

# 运行性能评估
result = evaluate_performance(
    array=ula,
    sources=sources,
    snr=10,           # 信噪比 10 dB
    n_snapshots=100,  # 快拍数
    n_monte_carlo=100, # Monte Carlo次数
    estimators=root_music,
    crb_types=['sto'],
    metrics=['mse', 'rmse'],
    verbose=1
)

print(result)
```

## 📚 示例教程

本项目提供了一系列Jupyter Notebook教程：

| 文件 | 内容 |
|------|------|
| [ex0_design_arrays.ipynb](examples/ex0_design_arrays.ipynb) | 阵列设计与可视化 |
| [ex1_doa_with_ula.ipynb](examples/ex1_doa_with_ula.ipynb) | 使用ULA进行DoA估计 |
| [ex2_sparse_array.ipynb](examples/ex2_sparse_array.ipynb) | 稀疏阵列与协方差重建 |
| [ex3_performance.ipynb](examples/ex3_performance.ipynb) | DoA算法性能评估 |
| [ex4_coherent_signal.ipynb](examples/ex4_coherent_signal.ipynb) | 相干信号源处理 |
| [ex5_covariance_reconstruction.ipynb](examples/ex5_covariance_reconstruction.ipynb) | 协方差矩阵重建算法 |
| [ex6_performance_sdp.ipynb](examples/ex6_performance_sdp.ipynb) | 基于SDP方法的性能分析 |

运行示例：

```bash
cd examples
jupyter notebook
```

## 📄 协议

本项目与原项目一样，遵循 **MIT 协议**。详情请见 [LICENSE](LICENSE) 文件。

原项目的版权归属于其原作者 [morriswmz](https://github.com/morriswmz)。本项目在其基础上修改所产生的版权，由本项目的贡献者所有。

## 🙏 致谢

衷心感谢原项目作者 [morriswmz](https://github.com/morriswmz) 的前期工作，为社区提供了宝贵的入门工具。

## 📖 参考资料

### 协方差矩阵重建算法

1. Z. Yang, L. Xie, and C. Zhang, "A Discretization-Free Sparse and Parametric Approach for Linear Array Signal Processing," *IEEE Transactions on Signal Processing*, vol. 62, no. 19, pp. 4959-4973, Oct. 2014.

2. C. Zhou, Y. Gu, X. Fan, Z. Shi, G. Mao, and Y. D. Zhang, "Direction-of-Arrival Estimation for Coprime Array via Virtual Array Interpolation," *IEEE Transactions on Signal Processing*, vol. 66, no. 22, pp. 5956-5971, Nov. 2018.

3. X. Wu, W.-P. Zhu, and J. Yan, "A Toeplitz Covariance Matrix Reconstruction Approach for Direction-of-Arrival Estimation," *IEEE Transactions on Vehicular Technology*, vol. 66, no. 9, pp. 8223-8237, Sept. 2017.

4. M. Wang, Z. Zhang, and A. Nehorai, "Grid-Less DOA Estimation Using Sparse Linear Arrays Based on Wasserstein Distance," *IEEE Signal Processing Letters*, vol. 26, no. 6, pp. 838-842, June 2019.

5. R. R. Pote and B. D. Rao, "Maximum Likelihood-Based Gridless DoA Estimation Using Structured Covariance Matrix Recovery and SBL With Grid Refinement," *IEEE Transactions on Signal Processing*, vol. 71, pp. 802-815, 2023.