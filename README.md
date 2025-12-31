# DoaTools (Revived & Enhanced)

> **这是一个基于原项目 [morriswmz/doatools.py](https://github.com/morriswmz/doatools.py) 的活跃维护分支。**
>
> 原项目为使用MIT协议的开源项目，但由于已长期未更新（最后维护于约2018年），本项目旨在对其进行现代化重构、功能增强和长期维护。

## 📖 项目起源

本项目派生（Fork）自 [morriswmz/doatools.py](https://github.com/morriswmz/doatools.py)，一个用于**波达方向（Direction of Arrival, DoA）估计**的Python工具库。

鉴于原项目已停止维护多年，本项目计划：
1.  **修复已知问题与兼容性**：适配新版Python及科学计算库（如NumPy、SciPy）。
2.  **代码重构与优化**：提升代码可读性、可维护性和计算效率。
3.  **扩展算法与功能**：计划引入更多现代DoA估计算法及实用功能。
4.  **完善文档与示例**：提供更清晰的使用说明和实际应用案例。

## ⚡ 主要变更与计划

*（请在此处列出您已经完成和计划进行的主要改动。例如：）*
- **[已完成]** 将代码适配至 Python 3.10+。
- **[已完成]** 使用 `numpy` 新API重构部分向量化计算。
- **[进行中]** 重构项目结构，增加模块化设计。
- **[计划中]** 添加 `MUSIC`、`ESPRIT` 算法的快速实现。
- **[计划中]** 增加对稀疏阵列和协处理的支持。
- **[计划中]** 完善自动化测试和持续集成。

## 🚀 快速开始

### 安装

```bash
# 从本仓库安装开发版本
pip install git+https://github.com/hengxt/doatools.git
```

### 基本用法

```python
import numpy as np
from doatools import 

# 您的示例代码
# ...
```

## 📄 协议

本项目与原项目一样，遵循 **MIT 协议**。详情请见 [LICENSE](LICENSE) 文件。

原项目的版权归属于其原作者 [morriswmz](https://github.com/morriswmz)。本项目在其基础上修改所产生的版权，由本项目的贡献者所有。

## 🙏 致谢

衷心感谢原项目作者 [morriswmz](https://github.com/morriswmz) 的前期工作，为社区提供了宝贵的入门工具。

---

### 下一步建议
您可以根据原项目的具体内容来完善此README：
1.  填充原项目的**简短介绍**和**核心功能**。
2.  在“快速开始”部分，补充原项目或您新版本的**典型用法示例**。
3.  明确列出您**已实现**和**将实现**的功能差异。
4.  更新安装指引中的仓库链接。

这样既能清晰地表明项目渊源，又能突出您的改进和维护状态。
