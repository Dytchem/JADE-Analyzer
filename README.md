# JADE-Analyzer

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

JADE-Analyzer 是一个用于分析 JADE-NAMD 激发态动力学模拟输出数据的 Python 库。

## 功能特性

- **多模块数据处理**: 状态、坐标、密度信息、能量、电子时间演化等
- **单/多轨迹支持**: `Single` 和 `Multi` 类处理不同规模的数据
- **统一接口**: 所有数据类继承自统一基类，便于集成
- **数据整合**: 支持多类型数据整合分析
- **可视化工具**: 内置状态分布、几何参数等可视化功能

## 项目结构

```
JADE-Analyzer/
├── state/          # 电子状态数据处理
├── coordinate/     # 原子坐标与几何参数
├── di/             # 密度信息 (Mulliken电荷、偶极矩)
├── energy/         # 能量数据处理
├── ele_time/       # 电子时间演化数据
├── hop_coord/      # 跃迁-坐标关联分析
├── unite/          # 数据整合模块
└── bylw/           # 示例脚本
```

## 安装

```bash
git clone https://github.com/yourusername/JADE-Analyzer.git
cd JADE-Analyzer
pip install -r requirements.txt
```

## 快速开始

### 1. 读取状态数据

```python
from state import StateMulti

state = StateMulti("path/to/trajectories", max_i_time=500)
print(state.description())
```

### 2. 读取坐标数据

```python
from coordinate import CoordMulti, Geometry

coord = CoordMulti(paths, max_i_time=500)
geometry = Geometry(coord)
dihedral = geometry.calculate_dihedral("C3", "N2", "N1", "C7")
```

### 3. 读取电子时间演化数据

```python
from ele_time import EleTimeSingle

ele = EleTimeSingle("path/to/ele_time.out")
populations = ele.get_populations()  # 获取态布居数
```

### 4. 可视化

```python
from state import PlotDistribution

plotter = PlotDistribution(state)
plotter.plot_normalized(save_path="state_dist.png")
```

## 核心模块

| 模块 | 功能 |
|------|------|
| `state` | 电子状态跃迁分析 |
| `coordinate` | 原子坐标与几何参数计算 |
| `di` | Mulliken电荷与偶极矩 |
| `energy` | 能量数据处理 |
| `ele_time` | 电子时间演化数据读取 |
| `hop_coord` | 跃迁与几何参数关联分析 |
| `unite` | 多数据源整合 |

## 运行示例

```bash
cd bylw/azb_E_dp
python run_state.py
```

## 许可证

MIT License
