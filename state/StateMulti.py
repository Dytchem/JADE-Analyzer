import numpy as np
import pandas as pd
from StateSingle import StateSingle


class StateMuti:
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
        return {"hop_time": hop_time, "crash_time": crash_time}


if __name__ == "__main__":
    path = [
        r"E:\GitHub\JADE-Analyzer\sample\1",
        r"E:\GitHub\JADE-Analyzer\sample\1_del",
        r"E:\GitHub\JADE-Analyzer\sample\2",
    ]
    max_i_time = 500
    state = StateMuti(path, max_i_time)
    state.save_to_csv(r"E:\GitHub\JADE-Analyzer\output\state_muti.csv")

    print(state.data)
    print(state.count_state())
    print(state.distribution_change())
