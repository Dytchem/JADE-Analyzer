import sys
from pathlib import Path

import numpy as np
import pandas as pd


CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from StateSingle import StateSingle


class StateMulti:
    def __init__(self, path, max_i_time, type="folder"):
        self.max_i_time = max_i_time
        if type == "folder":
            frames = []
            for i, p in enumerate(path):
                state_single = StateSingle(p, max_i_time, type="folder")
                if i == 0:
                    first = state_single.data.loc[:, ["time", "state"]].copy()
                    first.columns = ["time", "state_No.1"]
                    frames.append(first)
                else:
                    frames.append(
                        state_single.data.loc[:, ["state"]].rename(
                            columns={"state": f"state_No.{i+1}"}
                        )
                    )

            if not frames:
                raise ValueError("path is empty when type is 'folder'")
            self.data = pd.concat(frames, axis=1)

        elif type == "csv":
            self.data = pd.read_csv(path)
        elif type == "pickle":
            self.data = pd.read_pickle(path)
        else:
            raise ValueError("type must be 'folder' or 'csv' or 'pickle'")
        self.n = self.data.shape[1] - 1
        self.time_interval = self.data["time"][1] - self.data["time"][0]

    def set_time_series(self, time_series):
        time_array = np.asarray(time_series, dtype=float)
        if time_array.ndim != 1:
            raise ValueError("time_series must be a 1D sequence")
        if len(time_array) != len(self.data):
            raise ValueError(
                f"time_series length {len(time_array)} does not match data length {len(self.data)}"
            )

        self.data.loc[:, "time"] = time_array
        self.time_interval = (
            time_array[1] - time_array[0] if len(time_array) > 1 else np.nan
        )

    def save_to_csv(self, path):
        self.data.to_csv(path, index=False)

    def save_to_pickle(self, path):
        self.data.to_pickle(path)

    def count_state(self):
        ret = pd.DataFrame({"time": self.data["time"]})
        ret["count_state1"] = self.data.iloc[:, 1:].apply(
            lambda x: (x == 1).sum(), axis=1
        )
        ret["count_state2"] = self.data.iloc[:, 1:].apply(
            lambda x: (x == 2).sum(), axis=1
        )
        ret["count_crash"] = self.data.iloc[:, 1:].apply(
            lambda x: (x == 0).sum(), axis=1
        )

        return ret

    def distribution_change(self):
        hop_time = []  # 2->11111111(00000000)
        crash_time = []  # 2/1->000000000
        for i in range(self.n):
            if self.data.iloc[self.max_i_time, i + 1] == 2:
                hop_time.append(self.data.iloc[self.max_i_time, 0] + self.time_interval)
            elif self.data.iloc[self.max_i_time, i + 1] == 1:
                for j in range(self.max_i_time - 1, -1, -1):
                    if self.data.iloc[j, i + 1] == 2:
                        hop_time.append(self.data.iloc[j + 1, 0])
                        break
            else:
                flag = False  # 是否步入正常区
                for j in range(self.max_i_time - 1, -1, -1):
                    if self.data.iloc[j, i + 1] == 2:
                        if flag:
                            hop_time.append(self.data.iloc[j + 1, 0])
                        else:
                            crash_time.append(self.data.iloc[j + 1, 0])
                        break
                    elif self.data.iloc[j, i + 1] == 1:
                        if not flag:
                            crash_time.append(self.data.iloc[j + 1, 0])
                        flag = True
        hop_time.sort()
        crash_time.sort()
        return {"hop_time": hop_time, "crash_time": crash_time}

    def description(self):
        dc = self.distribution_change()
        hop_time = dc["hop_time"]

        # 最大时间内崩溃数量
        count_crash = len(dc["crash_time"])

        # 退激发前崩溃数量
        count_crash1 = self.n - len(hop_time)
        if len(hop_time) < self.n:
            hop_time += [self.data["time"].iloc[-1]] * count_crash

        # hop_time.sort()
        # print(hop_time)

        # 最大时间内未退激发的数量
        count_no_hop = sum(
            1 for i in range(self.n) if self.data.iloc[self.max_i_time, i + 1] == 2
        )

        # S0 和 S1 的数量相等的时间（退激发时间中位数）
        hop_time_median = np.median(hop_time)

        # 退激发时间平均数（未退激发按最大时间参与计算）
        hop_time_mean = np.mean(hop_time)

        # 退激发时间平均数（不算未退激发）
        hop_time_mean_exclude_no_hop = np.mean(
            [t for t in hop_time if t <= self.data["time"].iloc[-1]]
        )

        return {
            "运行过程崩溃数量": count_crash,
            "退激发前崩溃数量": count_crash1,
            "末尾未退激发数量": count_no_hop,
            "退激发时间中位数": hop_time_median,
            "退激发时间平均数": hop_time_mean,
            "退激发时间平均数（不算未退激发）": hop_time_mean_exclude_no_hop,
        }


if __name__ == "__main__":
    path = [f"E:\\GitHub\\JADE-Analyzer\\sample\\{i}" for i in range(1, 201)]
    max_i_time = 500
    state = StateMulti(path, max_i_time)
    state.save_to_csv(r"E:\GitHub\JADE-Analyzer\output\state_multi.csv")

    # print(state.data)
    # print(state.count_state())
    # print(state.distribution_change())
    print(state.description())
