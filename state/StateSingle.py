import os
from typing import Optional

import numpy as np
import pandas as pd

from unite.base import BaseData


class StateSingle(BaseData):
    """
    Single trajectory state data handler for JADE-NAMD simulations.
    
    Reads and processes state information from hop_all_time.out files,
    tracking electronic state transitions between S0 and S1 states.
    """
    
    def __init__(self, path: str, max_i_time: int, type: str = "folder"):
        """
        Initialize StateSingle.
        
        Args:
            path: Path to data source (folder with hop_all_time.out or CSV/Pickle file)
            max_i_time: Maximum time index
            type: Data source type ('folder', 'csv', or 'pickle')
        """
        self.max_i_time = max_i_time
        
        if type == "folder":
            time = np.zeros(max_i_time + 1)  # 时间序列，单位fs
            state = np.zeros(max_i_time + 1, dtype=int)  # 状态序列，1表示S0，2表示S1
            with open(os.path.join(path, "hop_all_time.out")) as f:
                for i in range(0, max_i_time + 1):
                    f.readline()
                    a = f.readline()
                    if not a:
                        break
                    a = a.strip().split()
                    time[i] = float(a[1])
                    state[i] = int(a[-1])
                    for j in range(8):
                        f.readline()
            # 时间序列补全
            time = np.arange(0, time[1] * (max_i_time + 0.5), time[1])
            data = pd.DataFrame({"time": time, "state": state})
        elif type == "csv":
            data = pd.read_csv(path)
        elif type == "pickle":
            data = pd.read_pickle(path)
        else:
            raise ValueError("type must be 'folder' or 'csv' or 'pickle'")
        
        super().__init__(data, max_i_time, "state")


if __name__ == "__main__":
    path = r"E:\GitHub\JADE-Analyzer\sample\1_del"
    max_i_time = 500
    state = StateSingle(path, max_i_time)
    state.save_to_csv(r"E:\GitHub\JADE-Analyzer\output\state_single.csv")

    print(state.data)
