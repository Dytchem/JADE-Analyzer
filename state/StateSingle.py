import os
import numpy as np
import pandas as pd


class StateSingle:
    def __init__(self, path, max_i_time, type="folder"):
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
                    # print(a)
                    time[i] = float(a[1])
                    state[i] = int(a[-1])
                    for j in range(8):
                        f.readline()
            # 时间序列补全
            time = np.arange(0, time[1] * (max_i_time + 0.5), time[1])
            self.data = pd.DataFrame({"time": time, "state": state})
        elif type == "csv":
            self.data = pd.read_csv(path)
        elif type == "pickle":
            self.data = pd.read_pickle(path)
        else:
            raise ValueError("type must be 'folder' or 'csv' or 'pickle'")
        self.time_interval = self.data["time"][1]

    def save_to_csv(self, path):
        self.data.to_csv(path, index=False)

    def save_to_pickle(self, path):
        self.data.to_pickle(path)


if __name__ == "__main__":
    path = r"E:\GitHub\JADE-Analyzer\sample\1_del"
    max_i_time = 500
    state = StateSingle(path, max_i_time)
    state.save_to_csv(r"E:\GitHub\JADE-Analyzer\output\state_single.csv")

    print(state.data)
