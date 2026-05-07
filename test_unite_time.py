"""
测试脚本：验证单轨迹和多轨迹的 Unite 功能，检查时间轴匹配情况
期望时间轴：0, 0.5, 1.0, 1.5, 2.0, ...
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd

from state import StateSingle, StateMulti
from coordinate import CoordSingle, CoordMulti
from energy import EnergySingle, EnergyMulti
from unite import DataUniter, MultiTrajectoryUniter


def test_single_trajectory_unite():
    """测试单轨迹 Unite 功能"""
    print("=" * 60)
    print("测试单轨迹 Unite 功能")
    print("=" * 60)
    
    # 加载数据
    state = StateSingle(r"E:\GitHub\JADE-Analyzer\sample\1", 500)
    coord = CoordSingle(r"E:\GitHub\JADE-Analyzer\sample\1", 500)
    energy = EnergySingle(r"E:\GitHub\JADE-Analyzer\sample\1", 500)
    
    print(f"State has_real_time: {state.has_real_time()}")
    print(f"Coord has_real_time: {coord.has_real_time()}")
    print(f"Energy has_real_time: {energy.has_real_time()}")
    
    # 创建 Uniter 并添加数据源
    uniter = DataUniter()
    uniter.add_source("state", state)
    uniter.add_source("coord", coord)
    uniter.add_source("energy", energy)
    
    # 获取 Unite 后的数据
    unified_data = uniter.get_unified_data()
    
    print(f"\nUnified data shape: {unified_data.shape}")
    print(f"Unified columns: {list(unified_data.columns)[:10]}... (total {len(unified_data.columns)})")
    
    # 检查时间轴
    time_values = unified_data['time'].values
    print(f"\n时间轴前10个值: {time_values[:10]}")
    print(f"时间轴后10个值: {time_values[-10:]}")
    
    # 验证时间轴是否正确
    expected_time = np.arange(0, 501, 1.0)
    time_diff = np.abs(time_values - expected_time)
    print(f"\n时间轴最大偏差: {np.max(time_diff):.6f}")
    print(f"时间轴是否匹配预期: {np.allclose(time_values, expected_time)}")
    
    return unified_data


def test_multi_trajectory_unite():
    """测试多轨迹 Unite 功能"""
    print("\n" + "=" * 60)
    print("测试多轨迹 Unite 功能")
    print("=" * 60)
    
    # 加载多轨迹数据
    paths = [
        r"E:\GitHub\JADE-Analyzer\sample\1",
        r"E:\GitHub\JADE-Analyzer\sample\2",
    ]
    
    state_multi = StateMulti(paths, 500)
    coord_multi = CoordMulti(paths, 500)
    energy_multi = EnergyMulti(paths, 500)
    
    print(f"StateMulti has_real_time: {state_multi.has_real_time()}")
    print(f"CoordMulti has_real_time: {coord_multi.has_real_time()}")
    print(f"EnergyMulti has_real_time: {energy_multi.has_real_time()}")
    
    # 创建 MultiTrajectoryUniter 并添加数据源
    multi_uniter = MultiTrajectoryUniter()
    multi_uniter.add_source("state", state_multi)
    multi_uniter.add_source("coord", coord_multi)
    multi_uniter.add_source("energy", energy_multi)
    
    # 检查每个轨迹的结果
    for i in range(1, multi_uniter.n_trajectories + 1):
        traj_uniter = multi_uniter.get_trajectory_uniter(i)
        unified_data = traj_uniter.get_unified_data()
        print(f"\n轨迹 {i}:")
        print(f"  数据形状: {unified_data.shape}")
        
        # 检查时间轴
        time_values = unified_data['time'].values
        print(f"  时间轴前5个值: {time_values[:5]}")
        print(f"  时间轴后5个值: {time_values[-5:]}")
        
        # 验证时间轴
        expected_time = np.arange(0, 501, 1.0)
        time_diff = np.abs(time_values - expected_time)
        print(f"  时间轴最大偏差: {np.max(time_diff):.6f}")
        print(f"  时间轴是否匹配预期: {np.allclose(time_values, expected_time)}")


def test_time_matching_with_different_sources():
    """测试不同数据源的时间匹配"""
    print("\n" + "=" * 60)
    print("测试不同数据源的时间匹配")
    print("=" * 60)
    
    state = StateSingle(r"E:\GitHub\JADE-Analyzer\sample\1", 500)
    coord = CoordSingle(r"E:\GitHub\JADE-Analyzer\sample\1", 500)
    
    print(f"State 时间轴: {state.data['time'].values[:5]}...")
    print(f"Coord 时间轴: {coord.data['time'].values[:5]}...")
    
    # 检查两者时间轴是否一致
    time_match = np.allclose(state.data['time'].values, coord.data['time'].values)
    print(f"\nState 和 Coord 时间轴是否一致: {time_match}")
    
    # 测试只有无真实时间数据源的 Unite
    coord2 = CoordSingle(r"E:\GitHub\JADE-Analyzer\sample\1", 500)
    uniter_no_real = DataUniter()
    uniter_no_real.add_source("coord", coord2)
    
    print(f"\n只有无真实时间数据源时:")
    print(f"  has_real_time: {uniter_no_real.has_real_time()}")
    print(f"  时间轴前5个值: {uniter_no_real.get_time_series()[:5]}")
    
    # 测试 set_time_series 方法
    custom_time = np.arange(0, 501, 1.0) * 0.5  # 0, 0.5, 1.0, 1.5, ...
    coord.set_time_series(custom_time)
    print(f"\n使用 set_time_series 设置自定义时间轴后:")
    print(f"Coord 时间轴: {coord.data['time'].values[:10]}")
    print(f"时间轴间隔: {coord.data['time'].values[1] - coord.data['time'].values[0]}")


if __name__ == "__main__":
    test_single_trajectory_unite()
    test_multi_trajectory_unite()
    test_time_matching_with_different_sources()
    
    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)