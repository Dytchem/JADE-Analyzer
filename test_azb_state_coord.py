"""
测试脚本：验证简化后的 step/time 设计
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


def test_basic_design():
    """测试基本设计"""
    print("=" * 60)
    print("测试简化后的 step/time 设计")
    print("=" * 60)
    
    sample_path = r"E:\GitHub\JADE-Analyzer\sample\1"
    
    # 加载 state 和 coord 数据
    state = StateSingle(sample_path, 500)
    coord = CoordSingle(sample_path, 500)
    
    print(f"\nState:")
    print(f"  has_real_time: {state.has_real_time()}")
    print(f"  data.shape: {state.data.shape}")
    print(f"  data.columns: {list(state.data.columns)}")
    print(f"  data.index[:5]: {state.data.index[:5].tolist()}")
    if 'time' in state.data.columns:
        print(f"  time values[:5]: {state.data['time'].values[:5]}")
    
    print(f"\nCoord:")
    print(f"  has_real_time: {coord.has_real_time()}")
    print(f"  data.shape: {coord.data.shape}")
    print(f"  data.columns[:5]: {list(coord.data.columns)[:5]}")
    print(f"  data.index[:5]: {coord.data.index[:5].tolist()}")


def test_unite():
    """测试合并功能"""
    print("\n" + "=" * 60)
    print("测试 State 和 Coord 合并")
    print("=" * 60)
    
    sample_path = r"E:\GitHub\JADE-Analyzer\sample\1"
    
    state = StateSingle(sample_path, 500)
    coord = CoordSingle(sample_path, 500)
    
    uniter = DataUniter()
    uniter.add_source("state", state)
    uniter.add_source("coord", coord)
    
    unified = uniter.get_unified_data()
    
    print(f"\n合并后数据:")
    print(f"  shape: {unified.shape}")
    print(f"  columns[:10]: {list(unified.columns)[:10]}")
    print(f"  index[:10]: {unified.index[:10].tolist()}")
    print(f"  time[:10]: {unified['time'].values[:10]}")
    
    print(f"\n验证:")
    print(f"  state_state 存在: {'state_state' in unified.columns}")
    print(f"  coord_N_1_x 存在: {'coord_N_1_x' in unified.columns}")


if __name__ == "__main__":
    test_basic_design()
    test_unite()
    
    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)