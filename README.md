# JADE-Analyzer

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

JADE-Analyzer 是一个用于分析 JADE-NAMD 激发态动力学模拟输出数据的 Python 库。它提供了一套完整的工具来处理、分析和可视化分子动力学模拟数据。

## 功能特性

- **多模块数据处理**: 支持状态(state)、坐标(coordinate)、密度信息(di)、能量(energy)等多种数据类型
- **单轨迹和多轨迹支持**: 提供 `Single` 和 `Multi` 类来处理单条或多条轨迹数据
- **统一的数据接口**: 所有数据类继承自统一的基类，便于数据集成和互操作
- **数据整合功能**: `unite` 模块支持将不同类型的数据整合在一起，共享时间序列信息
- **可视化工具**: 内置绘图功能，支持状态分布、几何参数等多种可视化

## 项目结构

```
JADE-Analyzer/
├── state/              # 电子状态数据处理模块
│   ├── StateSingle.py  # 单轨迹状态数据
│   ├── StateMulti.py   # 多轨迹状态数据
│   ├── PlotCount.py    # 状态计数可视化
│   └── PlotDistribution.py  # 状态分布可视化
├── coordinate/         # 原子坐标数据处理模块
│   ├── CoordSingle.py  # 单轨迹坐标数据
│   ├── CoordMulti.py   # 多轨迹坐标数据
│   ├── Geometry.py     # 几何参数计算
│   ├── PlotGeom.py     # 几何参数可视化
│   ├── CountGeom.py    # 几何参数统计
│   └── PlotCount.py    # 几何参数计数可视化
├── di/                 # 密度信息处理模块 (Mulliken电荷、偶极矩)
│   ├── DiSingle.py     # 单轨迹密度数据
│   └── DiMulti.py      # 多轨迹密度数据
├── energy/             # 能量数据处理模块
│   ├── EnergySingle.py # 单轨迹能量数据
│   └── EnergyMulti.py  # 多轨迹能量数据
├── hop_coord/          # 跃迁-坐标关联分析模块
│   ├── HopCoordSingle.py
│   ├── HopCoordMulti.py
│   └── _core.py        # 核心工具函数
├── unite/              # 数据整合模块
│   ├── base.py         # 基类定义
│   └── unite.py        # 数据整合类
├── font/               # 字体文件
└── bylw/               # 示例分析脚本
```

## 安装

### 依赖要求

- Python 3.8+
- numpy
- pandas
- matplotlib

### 安装方法

```bash
git clone https://github.com/yourusername/JADE-Analyzer.git
cd JADE-Analyzer
pip install -r requirements.txt
```

## 快速开始

### 1. 导入模块

```python
from state import StateSingle, StateMulti
from coordinate import CoordSingle, CoordMulti, Geometry
from di import DiSingle, DiMulti
from energy import EnergySingle, EnergyMulti
from unite import DataUniter, MultiTrajectoryUniter
```

### 2. 读取状态数据

```python
# 单轨迹
state_single = StateSingle("path/to/trajectory", max_i_time=500)
print(state_single.data.head())

# 多轨迹
paths = [f"sample/{i}" for i in range(1, 201)]
state_multi = StateMulti(paths, max_i_time=500)
print(state_multi.description())
```

### 3. 读取坐标数据并计算几何参数

```python
coord = CoordMulti(paths, max_i_time=500)
geometry = Geometry(coord)

# 计算二面角
dihedral = geometry.calculate_dihedral("C3", "N2", "N1", "C7")
print(dihedral.data.head())
```

### 4. 数据整合

```python
from unite import DataUniter

# 将状态和坐标数据整合
uniter = DataUniter(state_multi, coord)

# 访问整合后的数据
combined_data = uniter.get_combined_data()
```

### 5. 可视化

```python
# 状态分布可视化
from state import PlotDistribution

plotter = PlotDistribution(state_multi)
plotter.plot_normalized(save_path="state_distribution.png")

# 几何参数随时间变化
from coordinate import PlotGeom

geom_plotter = PlotGeom(geometry)
geom_plotter.plot(save_path="geometry_time.png")
```

## API 文档

### 数据类基类

#### BaseData

所有单轨迹数据类的基类，提供以下方法：

- `set_time_series(time_series)`: 设置时间序列
- `has_real_time()`: 检查是否包含真实时间值
- `get_time_series()`: 获取时间序列
- `save_to_csv(path)`: 保存为 CSV 文件
- `save_to_pickle(path)`: 保存为 Pickle 文件

#### BaseMultiData

所有多轨迹数据类的基类，继承自 `BaseData`，额外提供：

- `get_trajectory_columns(trajectory_idx)`: 获取指定轨迹的列名
- `get_trajectory_data(trajectory_idx)`: 获取指定轨迹的数据

### 状态模块 (state)

#### StateSingle

读取单轨迹状态数据：

```python
state = StateSingle(path, max_i_time, type="folder")  # 或 "csv", "pickle"
```

#### StateMulti

读取多轨迹状态数据：

```python
state = StateMulti(paths, max_i_time, type="folder")
```

**主要方法：**
- `count_state()`: 统计各状态数量
- `distribution_change()`: 计算状态跃迁时间分布
- `description(fit_max_time=None)`: 生成统计描述

### 坐标模块 (coordinate)

#### CoordSingle / CoordMulti

读取坐标数据。

#### Geometry

几何参数计算：

```python
geometry = Geometry(coord_data)

# 计算距离
geometry.calculate_distance(atom1, atom2)

# 计算角度
geometry.calculate_angle(atom1, atom2, atom3)

# 计算二面角
geometry.calculate_dihedral(atom1, atom2, atom3, atom4)
```

### 密度信息模块 (di)

#### DiSingle / DiMulti

读取 Mulliken 电荷和偶极矩数据：

```python
di = DiSingle(path, max_i_time)
print(di.data.columns)  # 包含 mulliken_* 和 dipole_* 列
```

### 能量模块 (energy)

#### EnergySingle / EnergyMulti

读取能量数据：

```python
energy = EnergySingle(path, max_i_time)
print(energy.data.columns)  # step, time, pot_ene, kin_ene, tot_ene, lin_mom_*
```

### 跃迁-坐标关联模块 (hop_coord)

#### HopCoordMulti

分析跃迁时间与几何参数的关联：

```python
hop_coord = HopCoordMulti(
    paths,
    max_i_time,
    ("C3", "N2", "N1", "C7"),  # 二面角原子
    ("N1", "N2", "C3"),          # 角度原子
    unwrap_dihedral=False
)

# 绘制跃迁时间与几何参数的散点图
hop_coord.plot_hop_vs("dihedral_C3-N2-N1-C7")
```

### 数据整合模块 (unite)

#### DataUniter

整合两个数据源：

```python
uniter = DataUniter(data1, data2)
combined = uniter.get_combined_data()
```

#### MultiTrajectoryUniter

整合多条轨迹数据：

```python
uniter = MultiTrajectoryUniter([state_multi, coord_multi])
```

## 示例脚本

项目包含 `bylw/` 文件夹，包含多个示例分析脚本：

- `run_state.py`: 状态数据分析
- `run_coordinate.py`: 坐标数据分析
- `run_di.py`: 密度信息分析
- `run_hop_coord.py`: 跃迁-坐标关联分析
- `run_statistics.py`: 统计分析

运行示例：

```bash
cd bylw/azb_E_dp
python run_state.py
```

## 输入文件格式

JADE-Analyzer 支持以下输入文件格式：

### 原始输出文件
- `hop_all_time.out`: 状态跃迁信息
- `traj_time.out`: 原子坐标轨迹
- `di_time.out`: 密度信息（Mulliken电荷、偶极矩）
- `energy_time.out`: 能量信息

### 导出格式
- CSV 文件
- Pickle 文件

## 贡献

欢迎提交 Issue 和 Pull Request！

## 许可证

MIT License

## 联系方式

如有问题或建议，请通过 GitHub Issues 联系。
