"""
测试脚本：验证 azb_E_dp 中 state 和 coord 的合并功能
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd

from state import StateSingle
from coordinate import CoordSingle
from unite import DataUniter


def test_state_coord_merge():
    """测试 state 和 coord 的合并功能"""
    print("=" * 60)
    print("测试 azb_E_dp 中 State 和 Coord 的合并")
    print("=" * 60)
    
    # 使用 sample 数据进行测试
    sample_path = r"E:\GitHub\JADE-Analyzer\sample\1"
    
    print(f"测试数据路径: {sample_path}")
    
    # 加载 state 和 coord 数据
    state = StateSingle(sample_path, 500)
    coord = CoordSingle(sample_path, 500)
    
    print(f"\nState:")
    print(f"  has_real_time: {state.has_real_time()}")
    print(f"  数据形状: {state.data.shape}")
    print(f"  列名: {list(state.data.columns)}")
    print(f"  step 前5个值: {state.data['step'].values[:5]}")
    if 'time' in state.data.columns:
        print(f"  time 前5个值: {state.data['time'].values[:5]}")
    
    print(f"\nCoord:")
    print(f"  has_real_time: {coord.has_real_time()}")
    print(f"  数据形状: {coord.data.shape}")
    print(f"  列名前5个: {list(coord.data.columns)[:5]}")
    print(f"  step 前5个值: {coord.data['step'].values[:5]}")
    
    # 创建 Uniter 并合并
    uniter = DataUniter()
    uniter.add_source("state", state)
    uniter.add_source("coord", coord)
    
    unified_data = uniter.get_unified_data()
    
    print(f"\n合并后数据:")
    print(f"  数据形状: {unified_data.shape}")
    print(f"  列名: {list(unified_data.columns)[:10]}...")
    print(f"  step 前10个值: {unified_data['step'].values[:10]}")
    print(f"  time 前10个值: {unified_data['time'].values[:10]}")
    
    # 验证时间轴
    expected_time = np.arange(0, 501, 1.0)
    time_match = np.allclose(unified_data['time'].values, expected_time)
    print(f"\n时间轴验证:")
    print(f"  预期时间轴: {expected_time[:5]}...")
    print(f"  实际时间轴: {unified_data['time'].values[:5]}...")
    print(f"  时间轴匹配: {time_match}")
    
    # 验证数据完整性
    print(f"\n数据完整性检查:")
    print(f"  state_state 列是否存在: {'state_state' in unified_data.columns}")
    print(f"  coord_N_1_x 列是否存在: {'coord_N_1_x' in unified_data.columns}")
    print(f"  数据行数: {len(unified_data)}")
    
    # 验证合并后 coord 是否获得了时间
    print(f"\nCoord 时间同步检查:")
    print(f"  coord.has_real_time() 仍为: {coord.has_real_time()}")
    print(f"  coord.get_time_series(): {coord.get_time_series()[:5] if coord.get_time_series() is not None else None}")


def test_multiple_energies():
    """测试多个不同能量的 trajectory 的合并"""
    print("\n" + "=" * 60)
    print("测试多个能量 trajectory 的合并")
    print("=" * 60)
    
    energies = ['E_dp=-0.0100', 'E_dp=0.0000', 'E_dp=0.0100']
    
    for energy in energies:
        print(f"\n处理 {energy}...")
        
        try:
            path = r"E:\GitHub\JADE-Analyzer\sample\1"
            
            state = StateSingle(path, 500)
            coord = CoordSingle(path, 500)
            
            uniter = DataUniter()
            uniter.add_source("state", state)
            uniter.add_source("coord", coord)
            
            unified = uniter.get_unified_data()
            
            print(f"  ✓ 合并成功")
            print(f"    数据形状: {unified.shape}")
            print(f"    step 长度: {len(unified['step'])}")
            print(f"    time 长度: {len(unified['time'])}")
            
        except Exception as e:
            print(f"  ✗ 合并失败: {e}")


if __name__ == "__main__":
    test_state_coord_merge()
    test_multiple_energies()
    
    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)